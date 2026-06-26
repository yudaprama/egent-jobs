// Package mediagen implements River workers for the media generation surface
// (image and video). Each worker replaces the corresponding fire-and-forget
// async call in lobehub/apps/server/src/routers/lambda/ by owning the full
// lifecycle: provider API call → download result → file+generation insert →
// async_tasks status update.
package mediagen

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"mime"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/riverqueue/river"

	"egent-jobs/asynctask"
)

// ---------- shared HTTP client ----------

var defaultHTTPClient = &http.Client{Timeout: 120 * time.Second}

// ---------- env helpers ----------

func mediaAPIKey() string {
	if v := os.Getenv("MEDIA_API_KEY"); v != "" {
		return v
	}
	return os.Getenv("MODEL_API_KEY")
}

func imageBaseURL() string {
	if v := os.Getenv("MEDIA_IMAGE_BASE_URL"); v != "" {
		return strings.TrimRight(v, "/")
	}
	if v := os.Getenv("MODEL_BASE_URL"); v != "" {
		return strings.TrimRight(v, "/")
	}
	return "https://api.openai.com/v1"
}

func videoBaseURL() string {
	if v := os.Getenv("MEDIA_VIDEO_BASE_URL"); v != "" {
		return strings.TrimRight(v, "/")
	}
	if v := os.Getenv("MODEL_BASE_URL"); v != "" {
		return strings.TrimRight(v, "/")
	}
	return "https://api.openai.com/v1"
}

// ====================================================================
// ImageGenerationWorker
// ====================================================================

// GenerateImageArgs is the River job payload. The producer (BFF lambda) creates
// these args from the resolved generation request and enqueues one job per
// generation row.
type GenerateImageArgs struct {
	TaskID            string `json:"taskId"`
	GenerationID      string `json:"generationId"`
	GenerationBatchID string `json:"generationBatchId"`
	UserID            string `json:"userId"`
	WorkspaceID       string `json:"workspaceId,omitempty"`
	Provider          string `json:"provider"`
	Model             string `json:"model"`
	Prompt            string `json:"prompt"`
	Width             int    `json:"width,omitempty"`
	Height            int    `json:"height,omitempty"`
	Seed              int    `json:"seed,omitempty"`
}

func (GenerateImageArgs) Kind() string { return "generate_image" }

// ImageGenerationWorker calls an OpenAI-compatible /v1/images/generations
// endpoint, downloads the resulting image, creates a file record, and updates
// the generations + async_tasks rows.
type ImageGenerationWorker struct {
	river.WorkerDefaults[GenerateImageArgs]

	pool    *pgxpool.Pool
	store   *asynctask.Store
	client  *http.Client
	logger  *slog.Logger
	timeout time.Duration
}

// ImageWorkerConfig configures the image generation worker.
type ImageWorkerConfig struct {
	Pool    *pgxpool.Pool
	Store   *asynctask.Store
	Client  *http.Client
	Logger  *slog.Logger
	Timeout time.Duration
}

func NewImageGenerationWorker(cfg ImageWorkerConfig) *ImageGenerationWorker {
	if cfg.Client == nil {
		cfg.Client = defaultHTTPClient
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 10 * time.Minute
	}
	return &ImageGenerationWorker{
		pool:    cfg.Pool,
		store:   cfg.Store,
		client:  cfg.Client,
		logger:  cfg.Logger,
		timeout: cfg.Timeout,
	}
}

func (w *ImageGenerationWorker) Work(ctx context.Context, job *river.Job[GenerateImageArgs]) error {
	log := w.logger.With(
		"job_id", job.ID,
		"kind", job.Kind,
		"attempt", job.Attempt,
		"task_id", job.Args.TaskID,
		"generation_id", job.Args.GenerationID,
	)
	ctx, cancel := context.WithTimeout(ctx, w.timeout)
	defer cancel()

	if err := w.store.MarkProcessing(ctx, job.Args.TaskID); err != nil {
		if pgx.ErrNoRows == err {
			log.Warn("async_tasks row missing; cancelling")
			return river.JobCancel(fmt.Errorf("asynctask %s not found: %w", job.Args.TaskID, err))
		}
		log.Error("mark processing failed", "error", err)
		return err
	}

	start := time.Now()

	imageURL, err := w.callImageGen(ctx, job.Args)
	if err != nil {
		return w.fail(ctx, job.Args.TaskID, asynctask.ErrorTypeGeneric, err, log)
	}
	log.Info("image generated", "image_url", truncateURL(imageURL))

	imgData, contentType, err := downloadBytes(ctx, w.client, imageURL)
	if err != nil {
		return w.fail(ctx, job.Args.TaskID, asynctask.ErrorTypeGeneric,
			fmt.Errorf("download image: %w", err), log)
	}
	ext := extensionForContentType(contentType)
	fileHash := fmt.Sprintf("%x", sha256.Sum256(imgData))
	fileName := fmt.Sprintf("generation_%s%s", job.Args.GenerationID, ext)
	log.Info("image downloaded", "bytes", len(imgData), "type", contentType, "hash", fileHash)

	if err := w.persistResult(ctx, job.Args, imgData, fileHash, fileName, contentType, imageURL); err != nil {
		return w.fail(ctx, job.Args.TaskID, asynctask.ErrorTypeGeneric,
			fmt.Errorf("persist result: %w", err), log)
	}

	if err := w.store.MarkSuccess(ctx, job.Args.TaskID, time.Since(start).Milliseconds()); err != nil {
		log.Error("mark success failed", "error", err)
		return err
	}
	log.Info("image generation succeeded", "duration_ms", time.Since(start).Milliseconds())
	return nil
}

func (w *ImageGenerationWorker) callImageGen(ctx context.Context, args GenerateImageArgs) (string, error) {
	baseURL := imageBaseURL()
	apiKey := mediaAPIKey()

	var size string
	if args.Width > 0 && args.Height > 0 {
		size = fmt.Sprintf("%dx%d", args.Width, args.Height)
	}

	body := map[string]any{
		"model":  args.Model,
		"prompt": args.Prompt,
		"n":      1,
	}
	if size != "" {
		body["size"] = size
	}

	raw, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/images/generations", bytes.NewReader(raw))
	if err != nil {
		return "", fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	resp, err := w.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("provider call: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("provider returned %d: %s", resp.StatusCode, truncateStr(string(respBody), 500))
	}

	var parsed struct {
		Data []struct {
			URL string `json:"url"`
		} `json:"data"`
	}
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return "", fmt.Errorf("decode response: %w", err)
	}
	if len(parsed.Data) == 0 || parsed.Data[0].URL == "" {
		return "", fmt.Errorf("provider returned 0 images")
	}
	return parsed.Data[0].URL, nil
}

func (w *ImageGenerationWorker) persistResult(ctx context.Context, args GenerateImageArgs, imgData []byte, fileHash, fileName, contentType, imageURL string) error {
	assetJSON := fmt.Sprintf(`{"url":"%s","type":"image"}`, imageURL)

	_, err := w.pool.Exec(ctx, `
		WITH inserted_file AS (
			INSERT INTO public.files (user_id, file_type, name, size, url, "source", file_hash)
			VALUES ($1, $2, $3, $4, $5, 'ImageGeneration', $6)
			RETURNING id
		)
		UPDATE public.generations
		   SET asset = $7::jsonb,
		       file_id = (SELECT id FROM inserted_file),
		       updated_at = NOW()
		 WHERE id = $8 AND user_id = $1`,
		args.UserID,
		contentType,
		fileName,
		len(imgData),
		imageURL,
		fileHash,
		assetJSON,
		args.GenerationID,
	)
	return err
}

func (w *ImageGenerationWorker) fail(ctx context.Context, taskID, errType string, err error, log *slog.Logger) error {
	log.Error("image generation failed", "error", err)
	if markErr := w.store.MarkError(ctx, taskID, errType, err.Error()); markErr != nil {
		log.Error("mark error failed", "error", markErr)
	}
	return err
}

// ====================================================================
// VideoGenerationWorker
// ====================================================================

// GenerateVideoArgs is the River job payload for video generation.
type GenerateVideoArgs struct {
	TaskID            string `json:"taskId"`
	GenerationID      string `json:"generationId"`
	GenerationBatchID string `json:"generationBatchId"`
	UserID            string `json:"userId"`
	WorkspaceID       string `json:"workspaceId,omitempty"`
	Provider          string `json:"provider"`
	Model             string `json:"model"`
	Prompt            string `json:"prompt"`
	ImageURL          string `json:"imageUrl,omitempty"`
	EndImageURL       string `json:"endImageUrl,omitempty"`
	Duration          int    `json:"duration,omitempty"`
	AspectRatio       string `json:"aspectRatio,omitempty"`
	Seed              int    `json:"seed,omitempty"`
}

func (GenerateVideoArgs) Kind() string { return "generate_video" }

// VideoGenerationWorker generates a video by calling the provider API, then
// polls for completion (or uses a returned URL directly), downloads the result,
// creates a file record, and updates the generations + async_tasks rows.
type VideoGenerationWorker struct {
	river.WorkerDefaults[GenerateVideoArgs]

	pool    *pgxpool.Pool
	store   *asynctask.Store
	client  *http.Client
	logger  *slog.Logger
	timeout time.Duration
}

// VideoWorkerConfig configures the video generation worker.
type VideoWorkerConfig struct {
	Pool    *pgxpool.Pool
	Store   *asynctask.Store
	Client  *http.Client
	Logger  *slog.Logger
	Timeout time.Duration
}

func NewVideoGenerationWorker(cfg VideoWorkerConfig) *VideoGenerationWorker {
	if cfg.Client == nil {
		cfg.Client = defaultHTTPClient
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 15 * time.Minute
	}
	return &VideoGenerationWorker{
		pool:    cfg.Pool,
		store:   cfg.Store,
		client:  cfg.Client,
		logger:  cfg.Logger,
		timeout: cfg.Timeout,
	}
}

func (w *VideoGenerationWorker) Work(ctx context.Context, job *river.Job[GenerateVideoArgs]) error {
	log := w.logger.With(
		"job_id", job.ID,
		"kind", job.Kind,
		"attempt", job.Attempt,
		"task_id", job.Args.TaskID,
		"generation_id", job.Args.GenerationID,
	)
	ctx, cancel := context.WithTimeout(ctx, w.timeout)
	defer cancel()

	if err := w.store.MarkProcessing(ctx, job.Args.TaskID); err != nil {
		if pgx.ErrNoRows == err {
			log.Warn("async_tasks row missing; cancelling")
			return river.JobCancel(fmt.Errorf("asynctask %s not found: %w", job.Args.TaskID, err))
		}
		log.Error("mark processing failed", "error", err)
		return err
	}

	start := time.Now()

	videoURL, err := w.generateVideo(ctx, job.Args)
	if err != nil {
		return w.failVid(ctx, job.Args.TaskID, err, log)
	}
	log.Info("video generated", "video_url", truncateURL(videoURL))

	vidData, contentType, err := downloadBytes(ctx, w.client, videoURL)
	if err != nil {
		return w.failVid(ctx, job.Args.TaskID,
			fmt.Errorf("download video: %w", err), log)
	}
	fileHash := fmt.Sprintf("%x", sha256.Sum256(vidData))
	ext := extensionForContentType(contentType)
	fileName := fmt.Sprintf("generation_%s%s", job.Args.GenerationID, ext)
	log.Info("video downloaded", "bytes", len(vidData), "type", contentType, "hash", fileHash)

	if err := w.persistVideoResult(ctx, job.Args, vidData, fileHash, fileName, contentType, videoURL, log); err != nil {
		return w.failVid(ctx, job.Args.TaskID,
			fmt.Errorf("persist result: %w", err), log)
	}

	if err := w.store.MarkSuccess(ctx, job.Args.TaskID, time.Since(start).Milliseconds()); err != nil {
		log.Error("mark success failed", "error", err)
		return err
	}
	log.Info("video generation succeeded", "duration_ms", time.Since(start).Milliseconds())
	return nil
}

func (w *VideoGenerationWorker) generateVideo(ctx context.Context, args GenerateVideoArgs) (string, error) {
	baseURL := videoBaseURL()
	apiKey := mediaAPIKey()

	params := map[string]any{
		"model":  args.Model,
		"prompt": args.Prompt,
	}
	if args.ImageURL != "" {
		params["image_url"] = args.ImageURL
	}
	if args.EndImageURL != "" {
		params["end_image_url"] = args.EndImageURL
	}
	if args.Duration > 0 {
		params["duration"] = args.Duration
	}
	if args.AspectRatio != "" {
		params["aspect_ratio"] = args.AspectRatio
	}
	if args.Seed > 0 {
		params["seed"] = args.Seed
	}

	body := map[string]any{
		"model":  args.Model,
		"params": params,
	}

	raw, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("marshal video request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/v1/video/generations", bytes.NewReader(raw))
	if err != nil {
		return "", fmt.Errorf("build video request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	resp, err := w.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("video provider call: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read video response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("video provider returned %d: %s", resp.StatusCode, truncateStr(string(respBody), 500))
	}

	// Some providers return the video URL directly.
	var direct struct {
		Data []struct {
			URL string `json:"url"`
		} `json:"data"`
	}
	if err := json.Unmarshal(respBody, &direct); err == nil && len(direct.Data) > 0 && direct.Data[0].URL != "" {
		return direct.Data[0].URL, nil
	}

	// Others return an inference ID for polling.
	var pollResponse struct {
		InferenceID string `json:"inferenceId"`
		ID          string `json:"id"`
	}
	if err := json.Unmarshal(respBody, &pollResponse); err != nil || (pollResponse.InferenceID == "" && pollResponse.ID == "") {
		return "", fmt.Errorf("unrecognized video provider response: %s", truncateStr(string(respBody), 500))
	}

	inferenceID := pollResponse.InferenceID
	if inferenceID == "" {
		inferenceID = pollResponse.ID
	}

	// Persist inference_id for observability.
	if _, err := w.pool.Exec(ctx, `
		UPDATE public.async_tasks SET inference_id = $2, updated_at = NOW() WHERE id = $1`,
		args.TaskID, inferenceID); err != nil {
		w.logger.Warn("failed to persist inference_id", "error", err)
	}

	pollURL := fmt.Sprintf("%s/v1/video/status/%s", baseURL, inferenceID)
	tick := time.NewTicker(5 * time.Second)
	defer tick.Stop()

	for attempts := 0; attempts < 120; attempts++ {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-tick.C:
		}

		videoURL, done, pollErr := w.pollStatus(ctx, pollURL, apiKey)
		if pollErr != nil {
			return "", fmt.Errorf("poll status: %w", pollErr)
		}
		if done {
			return videoURL, nil
		}
	}

	return "", fmt.Errorf("video generation timed out after %d attempts", 120)
}

func (w *VideoGenerationWorker) pollStatus(ctx context.Context, pollURL, apiKey string) (string, bool, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, pollURL, nil)
	if err != nil {
		return "", false, err
	}
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	resp, err := w.client.Do(req)
	if err != nil {
		return "", false, err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", false, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", false, fmt.Errorf("poll returned %d", resp.StatusCode)
	}

	var result struct {
		Status   string `json:"status"`
		VideoURL string `json:"videoUrl"`
		URL      string `json:"url"`
		Output   struct {
			VideoURL string `json:"videoUrl"`
			URL      string `json:"url"`
		} `json:"output"`
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return "", false, nil
	}

	switch result.Status {
	case "success", "completed", "succeeded":
		url := coalesce(result.VideoURL, result.URL, result.Output.VideoURL, result.Output.URL)
		if url == "" {
			return "", false, fmt.Errorf("success status but no video URL in response")
		}
		return url, true, nil
	case "error", "failed":
		return "", false, fmt.Errorf("video generation failed: %s", truncateStr(string(respBody), 300))
	default:
		return "", false, nil
	}
}

func (w *VideoGenerationWorker) persistVideoResult(ctx context.Context, args GenerateVideoArgs, vidData []byte, fileHash, fileName, contentType, videoURL string, log *slog.Logger) error {
	assetJSON := fmt.Sprintf(`{"url":"%s","type":"video"}`, videoURL)

	_, err := w.pool.Exec(ctx, `
		WITH inserted_file AS (
			INSERT INTO public.files (user_id, file_type, name, size, url, "source", file_hash)
			VALUES ($1, $2, $3, $4, $5, 'VideoGeneration', $6)
			RETURNING id
		)
		UPDATE public.generations
		   SET asset = $7::jsonb,
		       file_id = (SELECT id FROM inserted_file),
		       updated_at = NOW()
		 WHERE id = $8 AND user_id = $1`,
		args.UserID,
		contentType,
		fileName,
		len(vidData),
		videoURL,
		fileHash,
		assetJSON,
		args.GenerationID,
	)
	return err
}

func (w *VideoGenerationWorker) failVid(ctx context.Context, taskID string, err error, log *slog.Logger) error {
	log.Error("video generation failed", "error", err)
	if markErr := w.store.MarkError(ctx, taskID, asynctask.ErrorTypeGeneric, err.Error()); markErr != nil {
		log.Error("mark error failed", "error", markErr)
	}
	return err
}

// ====================================================================
// Utility functions
// ====================================================================

func downloadBytes(ctx context.Context, client *http.Client, url string) ([]byte, string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, "", err
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, "", fmt.Errorf("download returned %d", resp.StatusCode)
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, "", err
	}
	ct := resp.Header.Get("Content-Type")
	if ct == "" {
		ct = "application/octet-stream"
	}
	return data, ct, nil
}

func extensionForContentType(ct string) string {
	exts, err := mime.ExtensionsByType(ct)
	if err == nil && len(exts) > 0 {
		return exts[0]
	}
	switch {
	case strings.Contains(ct, "png"):
		return ".png"
	case strings.Contains(ct, "jpeg"), strings.Contains(ct, "jpg"):
		return ".jpg"
	case strings.Contains(ct, "gif"):
		return ".gif"
	case strings.Contains(ct, "webp"):
		return ".webp"
	case strings.Contains(ct, "mp4"), strings.Contains(ct, "mpeg"):
		return ".mp4"
	case strings.Contains(ct, "webm"):
		return ".webm"
	default:
		return ".bin"
	}
}

func truncateURL(u string) string {
	if idx := strings.Index(u, "?"); idx >= 0 {
		return u[:idx]
	}
	return u
}

func truncateStr(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

func coalesce(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

// Interface checks.
var _ river.Worker[GenerateImageArgs] = (*ImageGenerationWorker)(nil)
var _ river.Worker[GenerateVideoArgs] = (*VideoGenerationWorker)(nil)
