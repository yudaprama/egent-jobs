package memoryingest

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"regexp"
	"strings"
)

// Gatekeeper prompt + result. Mirrors the TS
// `packages/memory-user-memory/src/extractors/gatekeeper.ts` — a binary
// relevance check that runs before per-layer extraction to avoid
// burning LLM tokens on conversations that contain no durable user
// information.

const gatekeeperSystemPrompt = `You are a memory-extraction gatekeeper.

Decide whether the conversation below contains any durable, personal
information worth remembering about the user — identity, activities,
contexts, experiences, or preferences that would still matter days
or weeks from now.

Reply ONLY with a JSON object of shape:
{"relevant": true|false, "reason": "<one short sentence>"}

Reply false when the conversation is:
- greetings or small talk with no personal content
- technical questions unrelated to the user themselves
- ephemeral operational chatter (cron status, build logs)

Reply true when the conversation mentions:
- who the user is, their role, relationships, identity
- recurring or notable activities the user engages in
- projects, goals, or ongoing situations the user is tracking
- past experiences the user is reflecting on
- preferences, opinions, or stated likes/dislikes`

var gatekeeperJSONLine = regexp.MustCompile(`\{[^{}]*"relevant"[^{}]*\}`)

// gatekeeper asks the LLM whether the topic is worth extracting from.
// Returns (relevant, error). When the LLM reply is malformed the
// worker defaults to "relevant = true" so we err on the side of
// memory extraction (matches the LobeHub behavior).
func (w *IngestWorker) gatekeeper(ctx context.Context, content string, log *slog.Logger) (bool, error) {
	reply, err := w.llm.Chat(ctx, gatekeeperSystemPrompt, content)
	if err != nil {
		return false, fmt.Errorf("gatekeeper chat: %w", err)
	}

	match := gatekeeperJSONLine.FindString(reply)
	if match == "" {
		log.Warn("gatekeeper returned no JSON; defaulting to relevant", "reply", truncate(reply, 200))
		return true, nil
	}

	var out struct {
		Relevant bool   `json:"relevant"`
		Reason   string `json:"reason"`
	}
	if err := json.Unmarshal([]byte(match), &out); err != nil {
		log.Warn("gatekeeper JSON parse failed; defaulting to relevant", "error", err)
		return true, nil
	}
	if out.Relevant {
		log.Info("gatekeeper approved", "reason", out.Reason)
	} else {
		log.Info("gatekeeper rejected", "reason", out.Reason)
	}
	return out.Relevant, nil
}

// --- Identity ---

const identitySystemPrompt = `You extract user identities from conversation.

Reply ONLY with a JSON array. Each element describes one identity fact
about the user (a person, role, or relationship the user has mentioned
about themselves or someone they consistently reference).

Shape:
[
  {
    "description": "<one-sentence factual statement>",
    "type": "personal" | "professional" | "demographic",
    "role": "<the user's role, if any>",
    "relationship": "<their relationship to someone or something>",
    "episodicDate": "<ISO 8601 if this is tied to a specific date>",
    "tags": ["<short label>", ...]
  }
]

Return [] when the conversation contains no durable identity facts.`

// extractIdentity parses the LLM reply and returns 0..n identities.
// Malformed rows are dropped and logged.
func (w *IngestWorker) extractIdentity(ctx context.Context, content string, log *slog.Logger) ([]CreateIdentityInput, error) {
	reply, err := w.llm.Chat(ctx, identitySystemPrompt, content)
	if err != nil {
		return nil, fmt.Errorf("identity chat: %w", err)
	}
	rows, err := parseArrayReply[struct {
		Description  string   `json:"description"`
		Type         string   `json:"type,omitempty"`
		Role         string   `json:"role,omitempty"`
		Relationship string   `json:"relationship,omitempty"`
		EpisodicDate string   `json:"episodicDate,omitempty"`
		Tags         []string `json:"tags,omitempty"`
	}](reply)
	if err != nil {
		log.Warn("identity parse failed", "error", err, "reply", truncate(reply, 200))
		return nil, nil
	}
	out := make([]CreateIdentityInput, 0, len(rows))
	for _, r := range rows {
		if strings.TrimSpace(r.Description) == "" {
			continue
		}
		out = append(out, CreateIdentityInput{
			Description:  r.Description,
			Type:         r.Type,
			Role:         r.Role,
			Relationship: r.Relationship,
			EpisodicDate: stringPtr(r.EpisodicDate),
			Tags:         r.Tags,
		})
	}
	return out, nil
}

// --- Activity ---

const activitySystemPrompt = `You extract user activities from conversation.

Reply ONLY with a JSON array. Each element describes one activity the
user mentioned doing or planning to do — anything the user is engaging
in that has a clear action.

Shape:
[
  {
    "type": "<verb or short category like 'running' | 'writing' | 'debugging'>",
    "narrative": "<one sentence describing what happened>",
    "notes": "<optional longer context>",
    "status": "pending" | "active" | "completed" | "cancelled",
    "tags": ["<short label>", ...]
  }
]

Return [] when the conversation contains no durable activities.`

func (w *IngestWorker) extractActivity(ctx context.Context, content string, log *slog.Logger) ([]CreateActivityInput, error) {
	reply, err := w.llm.Chat(ctx, activitySystemPrompt, content)
	if err != nil {
		return nil, fmt.Errorf("activity chat: %w", err)
	}
	rows, err := parseArrayReply[struct {
		Type      string   `json:"type"`
		Narrative string   `json:"narrative,omitempty"`
		Notes     string   `json:"notes,omitempty"`
		Status    string   `json:"status,omitempty"`
		Tags      []string `json:"tags,omitempty"`
	}](reply)
	if err != nil {
		log.Warn("activity parse failed", "error", err)
		return nil, nil
	}
	out := make([]CreateActivityInput, 0, len(rows))
	for _, r := range rows {
		if strings.TrimSpace(r.Type) == "" && strings.TrimSpace(r.Narrative) == "" {
			continue
		}
		out = append(out, CreateActivityInput{
			Type:      r.Type,
			Narrative: r.Narrative,
			Notes:     r.Notes,
			Status:    r.Status,
			Tags:      r.Tags,
		})
	}
	return out, nil
}

// --- Context ---

const contextSystemPrompt = `You extract ongoing contexts from conversation.

Reply ONLY with a JSON array. Each element describes one ongoing
situation, project, goal, or recurring theme the user is tracking.

Shape:
[
  {
    "title": "<short title>",
    "description": "<one sentence summary>",
    "type": "<optional category like 'project' | 'goal' | 'situation'>",
    "tags": ["<short label>", ...]
  }
]

Return [] when the conversation contains no durable contexts.`

func (w *IngestWorker) extractContext(ctx context.Context, content string, log *slog.Logger) ([]CreateContextInput, error) {
	reply, err := w.llm.Chat(ctx, contextSystemPrompt, content)
	if err != nil {
		return nil, fmt.Errorf("context chat: %w", err)
	}
	rows, err := parseArrayReply[struct {
		Title       string   `json:"title"`
		Description string   `json:"description,omitempty"`
		Type        string   `json:"type,omitempty"`
		Tags        []string `json:"tags,omitempty"`
	}](reply)
	if err != nil {
		log.Warn("context parse failed", "error", err)
		return nil, nil
	}
	out := make([]CreateContextInput, 0, len(rows))
	for _, r := range rows {
		if strings.TrimSpace(r.Title) == "" {
			continue
		}
		out = append(out, CreateContextInput{
			Title:       r.Title,
			Description: r.Description,
			Type:        r.Type,
			Tags:        r.Tags,
		})
	}
	return out, nil
}

// --- Experience ---

const experienceSystemPrompt = `You extract user experiences from conversation.

Reply ONLY with a JSON array. Each element describes one past event
the user went through (situation → action → key learning).

Shape:
[
  {
    "situation": "<what was happening>",
    "action": "<what the user did>",
    "keyLearning": "<what the user took away, optional>",
    "type": "<optional category like 'work' | 'personal'>",
    "tags": ["<short label>", ...]
  }
]

Return [] when the conversation contains no notable past experiences.`

func (w *IngestWorker) extractExperience(ctx context.Context, content string, log *slog.Logger) ([]CreateExperienceInput, error) {
	reply, err := w.llm.Chat(ctx, experienceSystemPrompt, content)
	if err != nil {
		return nil, fmt.Errorf("experience chat: %w", err)
	}
	rows, err := parseArrayReply[struct {
		Situation   string   `json:"situation,omitempty"`
		Action      string   `json:"action,omitempty"`
		KeyLearning string   `json:"keyLearning,omitempty"`
		Type        string   `json:"type,omitempty"`
		Tags        []string `json:"tags,omitempty"`
	}](reply)
	if err != nil {
		log.Warn("experience parse failed", "error", err)
		return nil, nil
	}
	out := make([]CreateExperienceInput, 0, len(rows))
	for _, r := range rows {
		if strings.TrimSpace(r.Situation) == "" && strings.TrimSpace(r.Action) == "" {
			continue
		}
		out = append(out, CreateExperienceInput{
			Situation:   r.Situation,
			Action:      r.Action,
			KeyLearning: r.KeyLearning,
			Type:        r.Type,
			Tags:        r.Tags,
		})
	}
	return out, nil
}

// --- Preference ---

const preferenceSystemPrompt = `You extract user preferences from conversation.

Reply ONLY with a JSON array. Each element describes one preference,
opinion, like, dislike, or directive the user has stated.

Shape:
[
  {
    "suggestions": "<the preference, phrased as a suggestion or directive>",
    "type": "<optional category like 'ui' | 'food' | 'work'>",
    "tags": ["<short label>", ...]
  }
]

Return [] when the conversation contains no preferences.`

func (w *IngestWorker) extractPreference(ctx context.Context, content string, log *slog.Logger) ([]CreatePreferenceInput, error) {
	reply, err := w.llm.Chat(ctx, preferenceSystemPrompt, content)
	if err != nil {
		return nil, fmt.Errorf("preference chat: %w", err)
	}
	rows, err := parseArrayReply[struct {
		Suggestions string   `json:"suggestions,omitempty"`
		Type        string   `json:"type,omitempty"`
		Tags        []string `json:"tags,omitempty"`
	}](reply)
	if err != nil {
		log.Warn("preference parse failed", "error", err)
		return nil, nil
	}
	out := make([]CreatePreferenceInput, 0, len(rows))
	for _, r := range rows {
		if strings.TrimSpace(r.Suggestions) == "" {
			continue
		}
		out = append(out, CreatePreferenceInput{
			Suggestions: r.Suggestions,
			Type:        r.Type,
			Tags:        r.Tags,
		})
	}
	return out, nil
}

// --- helpers ---

// arrayJSONLine finds the first JSON array in a reply. The LLM is
// told to reply with JSON only, but sometimes wraps the array in
// prose or markdown code fences. The regex tolerates that.
var arrayJSONLine = regexp.MustCompile(`(?s)\[[^\[\]]*\]`)

func parseArrayReply[T any](reply string) ([]T, error) {
	match := arrayJSONLine.FindString(reply)
	if match == "" {
		return nil, nil
	}
	var out []T
	if err := json.Unmarshal([]byte(match), &out); err != nil {
		return nil, fmt.Errorf("decode array: %w", err)
	}
	return out, nil
}

func stringPtr(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}