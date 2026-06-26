package fileingest

import "errors"

// errStorageNoSuchKey mirrors the TS handler's special-case when the upstream
// storage returns NoSuchKey. The worker records the task as Error and does
// NOT retry — re-uploading or fixing IAM is the only recourse.
var errStorageNoSuchKey = errors.New("fileingest: file content unavailable in storage (NoSuchKey)")

// IsStorageNoSuchKey is exported so the worker / tests can detect the sentinel.
func IsStorageNoSuchKey(err error) bool { return errors.Is(err, errStorageNoSuchKey) }
