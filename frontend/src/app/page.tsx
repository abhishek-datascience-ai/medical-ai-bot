"use client";

import { FormEvent, useMemo, useState } from "react";

type DemoUser = {
  username: string;
  password: string;
  role: string;
  label: string;
};

type LoginResponse = {
  access_token: string;
  token_type: string;
  username: string;
  role: string;
};

type SourceCitation = {
  source_document: string;
  section_title: string;
  collection: string;
};

type ChatResponse = {
  answer: string;
  sources: SourceCitation[];
  retrieval_type: string;
  role: string;
};

type CollectionsResponse = {
  role: string;
  collections: string[];
};

const demoUsers: DemoUser[] = [
  {
    username: "dr.mehta",
    password: "doctor",
    role: "doctor",
    label: "Dr. Mehta — Doctor",
  },
  {
    username: "nurse.priya",
    password: "nurse",
    role: "nurse",
    label: "Nurse Priya — Nurse",
  },
  {
    username: "billing.ravi",
    password: "billing_executive",
    role: "billing_executive",
    label: "Billing Ravi — Billing Executive",
  },
  {
    username: "tech.anand",
    password: "technician",
    role: "technician",
    label: "Tech Anand — Technician",
  },
  {
    username: "admin.sys",
    password: "admin",
    role: "admin",
    label: "Admin Sys — Admin",
  },
];

const sampleQuestions = [
  "How many claims are rejected?",
  "Show claim status distribution",
  "What drug formulary guidance is available for antibiotic use?",
  "What infection control guidance should nurses follow?",
  "What equipment maintenance guidance is available?",
  "Show me insurance billing codes",
];

function getApiBaseUrl(): string {
  return process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";
}

function formatRetrievalType(value: string): string {
  if (value === "sql_rag") {
    return "SQL RAG";
  }

  if (value === "hybrid_rag") {
    return "Hybrid RAG";
  }

  if (value === "blocked") {
    return "Access Blocked";
  }

  return value;
}

function formatRole(value: string): string {
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export default function Home() {
  const apiBaseUrl = useMemo(() => getApiBaseUrl(), []);

  const [selectedUsername, setSelectedUsername] = useState(
    demoUsers[0].username,
  );
  const [loginData, setLoginData] = useState<LoginResponse | null>(null);
  const [collections, setCollections] = useState<string[]>([]);
  const [question, setQuestion] = useState(sampleQuestions[0]);
  const [chatResponse, setChatResponse] = useState<ChatResponse | null>(null);
  const [isLoggingIn, setIsLoggingIn] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const selectedUser = demoUsers.find(
    (user) => user.username === selectedUsername,
  );

  async function handleLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!selectedUser) {
      setErrorMessage("Please select a demo user.");
      return;
    }

    setIsLoggingIn(true);
    setErrorMessage("");
    setChatResponse(null);
    setCollections([]);

    try {
      const loginResponse = await fetch(`${apiBaseUrl}/login`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          username: selectedUser.username,
          password: selectedUser.password,
        }),
      });

      if (!loginResponse.ok) {
        const payload = await loginResponse.json();
        throw new Error(payload.detail || "Login failed.");
      }

      const loginPayload = (await loginResponse.json()) as LoginResponse;
      setLoginData(loginPayload);

      const collectionsResponse = await fetch(
        `${apiBaseUrl}/collections/${loginPayload.role}`,
      );

      if (!collectionsResponse.ok) {
        const payload = await collectionsResponse.json();
        throw new Error(payload.detail || "Unable to load collections.");
      }

      const collectionsPayload =
        (await collectionsResponse.json()) as CollectionsResponse;

      setCollections(collectionsPayload.collections);
    } catch (error) {
      setLoginData(null);
      setCollections([]);
      setErrorMessage(
        error instanceof Error ? error.message : "Unexpected login error.",
      );
    } finally {
      setIsLoggingIn(false);
    }
  }

  async function handleChat(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!loginData) {
      setErrorMessage("Please log in before asking a question.");
      return;
    }

    const trimmedQuestion = question.trim();

    if (!trimmedQuestion) {
      setErrorMessage("Please enter a question.");
      return;
    }

    setIsSending(true);
    setErrorMessage("");
    setChatResponse(null);

    try {
      const response = await fetch(`${apiBaseUrl}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: trimmedQuestion,
          access_token: loginData.access_token,
        }),
      });

      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.detail || "Chat request failed.");
      }

      const payload = (await response.json()) as ChatResponse;
      setChatResponse(payload);
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Unexpected chat error.",
      );
    } finally {
      setIsSending(false);
    }
  }

  return (
    <main className="page-shell">
      <section className="hero-card">
        <div>
          <p className="eyebrow">Role-aware Healthcare Assistant</p>
          <h1>Medical-AI Bot</h1>
          <p className="hero-text">
            Secure RAG demo with role-based access, hybrid document retrieval,
            SQL analytics, Gemini answer generation, and source citations.
          </p>
        </div>

        <div className="api-status">
          <span>API</span>
          <strong>{apiBaseUrl}</strong>
        </div>
      </section>

      <section className="grid-layout">
        <aside className="panel">
          <h2>Login</h2>
          <p className="muted">
            Select a demo account to test role-specific access.
          </p>

          <form onSubmit={handleLogin} className="form-stack">
            <label htmlFor="demo-user">Demo user</label>
            <select
              id="demo-user"
              value={selectedUsername}
              onChange={(event) => setSelectedUsername(event.target.value)}
            >
              {demoUsers.map((user) => (
                <option key={user.username} value={user.username}>
                  {user.label}
                </option>
              ))}
            </select>

            <button type="submit" disabled={isLoggingIn}>
              {isLoggingIn ? "Logging in..." : "Login"}
            </button>
          </form>

          {loginData ? (
            <div className="session-card">
              <div>
                <span className="label">Active user</span>
                <strong>{loginData.username}</strong>
              </div>

              <div>
                <span className="label">Role</span>
                <strong className="role-badge">
                  {formatRole(loginData.role)}
                </strong>
              </div>

              <div>
                <span className="label">Accessible collections</span>
                <div className="collection-list">
                  {collections.map((collection) => (
                    <span key={collection}>{collection}</span>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="empty-card">
              No active session. Login to start asking questions.
            </div>
          )}
        </aside>

        <section className="panel chat-panel">
          <h2>Chat</h2>
          <p className="muted">
            Ask document questions or SQL analytics questions. Access is
            controlled by the logged-in role.
          </p>

          <div className="sample-grid">
            {sampleQuestions.map((sampleQuestion) => (
              <button
                key={sampleQuestion}
                type="button"
                className="sample-button"
                onClick={() => setQuestion(sampleQuestion)}
              >
                {sampleQuestion}
              </button>
            ))}
          </div>

          <form onSubmit={handleChat} className="chat-form">
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask a question..."
              rows={5}
            />

            <button type="submit" disabled={isSending || !loginData}>
              {isSending ? "Generating answer..." : "Ask"}
            </button>
          </form>

          {errorMessage ? (
            <div className="error-card">{errorMessage}</div>
          ) : null}

          {chatResponse ? (
            <article
              className={
                chatResponse.retrieval_type === "blocked"
                  ? "answer-card blocked-card"
                  : "answer-card"
              }
            >
              <div className="answer-header">
                <span className="retrieval-badge">
                  {formatRetrievalType(chatResponse.retrieval_type)}
                </span>
                <span className="role-small">
                  {formatRole(chatResponse.role)}
                </span>
              </div>

              <h3>Answer</h3>
              <p className="answer-text">{chatResponse.answer}</p>

              <h3>Sources</h3>
              {chatResponse.sources.length > 0 ? (
                <div className="source-list">
                  {chatResponse.sources.map((source, index) => (
                    <div
                      key={`${source.source_document}-${source.section_title}-${index}`}
                      className="source-card"
                    >
                      <strong>{source.source_document}</strong>
                      <span>{source.section_title}</span>
                      <em>{source.collection}</em>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="muted">No document sources returned.</p>
              )}
            </article>
          ) : null}
        </section>
      </section>
    </main>
  );
}
