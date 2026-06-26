package queue

import "testing"

func TestSanitizeDSN(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"", ""},
		{"no-scheme", "no-scheme"},
		{"postgres://onlyhost", "postgres://onlyhost"},
		{"postgres://user:pw@host:5432/db", "postgres://user:***@host:5432/db"},
		{"postgres://u@host:5432/db", "postgres://u@host:5432/db"},
	}
	for _, c := range cases {
		got := SanitizeDSN(c.in)
		if got != c.want {
			t.Errorf("SanitizeDSN(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}
