package host

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestParseRenderURLs(t *testing.T) {
	urls, err := ParseRenderURLs(" a/m=http://r1:8000/|https://r2 , b=http://r3 ,")
	require.NoError(t, err)
	require.Equal(t, map[string][]string{"a/m": {"http://r1:8000", "https://r2"}, "b": {"http://r3"}}, urls)

	urls, err = ParseRenderURLs("")
	require.NoError(t, err)
	require.Empty(t, urls)

	for _, bad := range []string{"http://r1", "=http://r1", "m=", "m=r1:8000", "m=http://r1|"} {
		_, err := ParseRenderURLs(bad)
		require.Error(t, err, bad)
	}
}

// renderServer answers /v1/chat/completions/render with one token per message
// plus one for a generation prompt, and records the bodies it was sent.
func renderServer(t *testing.T, status *atomic.Int32, bodies chan<- map[string]any) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/chat/completions/render", r.URL.Path)
		var body map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		if bodies != nil {
			bodies <- body
		}
		if code := int(status.Load()); code != 0 {
			w.WriteHeader(code)
			return
		}
		messages := body["messages"].([]any)
		n := len(messages)
		if body["add_generation_prompt"] != false {
			n++
		}
		if turn := messages[len(messages)-1].(map[string]any); turn["role"] == "assistant" {
			n += len(turn["content"].(string))
		}
		ids := make([]int, n)
		_ = json.NewEncoder(w).Encode(map[string]any{"token_ids": ids})
	}))
	t.Cleanup(srv.Close)
	return srv
}

// Test flow:
//  1. Count a prompt and a turn on a render server that records the bodies.
//  2. Require the output limits and stream fields dropped, the turn counted as the request with the turn less the
//     request with an empty turn, and both turn bodies without a generation prompt.
func TestRenderClientCountsTheTurnAgainstAnEmptyOne(t *testing.T) {
	var ok atomic.Int32
	bodies := make(chan map[string]any, 3)
	srv := renderServer(t, &ok, bodies)
	client := NewRenderClient(map[string][]string{"m": {srv.URL}}, nil)
	request := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":true},"max_tokens":9,"max_completion_tokens":9,"min_tokens":2,"logprobs_mode":"raw","continue_final_message":false}`)

	n, source, err := client.CountPrompt(context.Background(), "m", request)
	require.NoError(t, err)
	require.Equal(t, uint64(2), n)
	require.Equal(t, srv.URL, source)
	prompt := <-bodies
	for _, field := range []string{"stream", "stream_options", "max_tokens", "max_completion_tokens", "min_tokens", "logprobs_mode"} {
		require.NotContains(t, prompt, field)
	}

	n, _, err = client.CountTurn(context.Background(), "m", request, map[string]any{"role": "assistant", "content": "abcde"})
	require.NoError(t, err)
	require.Equal(t, uint64(5), n)
	for range 2 {
		body := <-bodies
		require.Equal(t, false, body["add_generation_prompt"])
		require.NotContains(t, body, "continue_final_message")
		require.Len(t, body["messages"], 2)
	}
}

// Two servers may use different chat templates. Both halves of a turn count must use one server.
func TestRenderClientCountsTurnOnOneServer(t *testing.T) {
	var calls [2]atomic.Int32
	servers := make([]*httptest.Server, 2)
	for i := range servers {
		index := i
		servers[i] = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			calls[index].Add(1)
			var body map[string]any
			require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
			messages := body["messages"].([]any)
			count := 2 + 100*index
			if turn := messages[len(messages)-1].(map[string]any); turn["role"] == "assistant" {
				count += len(turn["content"].(string))
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"token_ids": make([]int, count)})
		}))
		t.Cleanup(servers[i].Close)
	}
	client := NewRenderClient(map[string][]string{"m": {servers[0].URL, servers[1].URL}}, nil)
	request := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`)

	count, source, err := client.CountTurn(context.Background(), "m", request, map[string]any{"role": "assistant", "content": "abcde"})
	require.NoError(t, err)
	require.Equal(t, uint64(5), count)
	require.Equal(t, servers[0].URL, source)
	require.Equal(t, int32(2), calls[0].Load())
	require.Equal(t, int32(0), calls[1].Load())
}

// Test flow:
//  1. Count on two render servers, the first failing, then refusing the body.
//  2. Require a failure to move on to the next server and a refusal to stop at once.
func TestRenderClientMovesOnFromAFailingServer(t *testing.T) {
	var failing, healthy atomic.Int32
	failing.Store(http.StatusServiceUnavailable)
	first := renderServer(t, &failing, nil)
	second := renderServer(t, &healthy, nil)
	client := NewRenderClient(map[string][]string{"m": {first.URL, second.URL}}, nil)
	request := []byte(`{"messages":[{"role":"user","content":"hi"}]}`)
	for range 3 {
		n, _, err := client.CountPrompt(context.Background(), "m", request)
		require.NoError(t, err)
		require.Equal(t, uint64(2), n)
	}

	failing.Store(http.StatusBadRequest)
	healthy.Store(http.StatusBadRequest)
	_, _, err := client.CountPrompt(context.Background(), "m", request)
	require.True(t, errors.Is(err, errRenderRefused), "%v", err)

	failing.Store(http.StatusTooManyRequests)
	healthy.Store(http.StatusTooManyRequests)
	_, _, err = client.CountPrompt(context.Background(), "m", request)
	require.Error(t, err)
	require.False(t, errors.Is(err, errRenderRefused), "a voter's own limits are not the body's refusal")

	healthy.Store(http.StatusNotFound)
	failing.Store(http.StatusNotFound)
	_, _, err = client.CountPrompt(context.Background(), "m", request)
	require.True(t, errors.Is(err, errRenderModelMissing), "%v", err)

	_, _, err = client.CountPrompt(context.Background(), "other", request)
	require.True(t, errors.Is(err, errNoRenderServer), "%v", err)
	require.False(t, client.Serves("other"))
	require.True(t, client.Serves("m"))
}

// Test flow:
//  1. Count with an ML node and a render server for the model while the node counts, has no render route,
//     refuses the body, and is busy.
//  2. Require the node's count first; a missing route, a refusal or a busy node moving on to the render server;
//     and a busy node with no render server that counts reported as busy, so the vote is tried again.
func TestRenderClientCountsOnTheMLNodeFirst(t *testing.T) {
	var serverStatus atomic.Int32
	srv := renderServer(t, &serverStatus, nil)
	request := []byte(`{"messages":[{"role":"user","content":"hi"}]}`)
	var nodeStatus atomic.Int32
	var nodeBusy atomic.Bool
	node := func(_ context.Context, model, path string, body []byte) (int, []byte, bool, error) {
		require.Equal(t, "m", model)
		require.Equal(t, "/v1/chat/completions/render", path)
		if nodeBusy.Load() {
			return 0, nil, true, errors.New("no free nodes")
		}
		return int(nodeStatus.Load()), []byte(`{"token_ids":[1,2,3]}`), false, nil
	}
	client := NewRenderClient(map[string][]string{"m": {srv.URL}}, node)
	require.True(t, client.Serves("other"), "the ML node counts for any model it serves")

	nodeStatus.Store(http.StatusOK)
	n, source, err := client.CountPrompt(context.Background(), "m", request)
	require.NoError(t, err)
	require.Equal(t, uint64(3), n)
	require.Equal(t, RenderSourceMLNode, source)

	for _, status := range []int32{http.StatusNotFound, http.StatusBadRequest, http.StatusServiceUnavailable} {
		nodeStatus.Store(status)
		n, source, err = client.CountPrompt(context.Background(), "m", request)
		require.NoError(t, err, "node status %d", status)
		require.Equal(t, uint64(2), n)
		require.Equal(t, srv.URL, source)
	}

	nodeBusy.Store(true)
	serverStatus.Store(http.StatusServiceUnavailable)
	_, _, err = client.CountPrompt(context.Background(), "m", request)
	require.True(t, errors.Is(err, errRenderBusy), "%v", err)

	_, source, err = NewRenderClient(nil, node).CountPrompt(context.Background(), "m", request)
	require.True(t, errors.Is(err, errRenderBusy), "%v", err)
	require.Equal(t, RenderSourceMLNode, source)

	nodeBusy.Store(false)
	nodeStatus.Store(http.StatusUnprocessableEntity)
	serverStatus.Store(http.StatusBadRequest)
	_, _, err = client.CountPrompt(context.Background(), "m", request)
	require.True(t, errors.Is(err, errRenderRefused), "refused by the node and the server: %v", err)
}
