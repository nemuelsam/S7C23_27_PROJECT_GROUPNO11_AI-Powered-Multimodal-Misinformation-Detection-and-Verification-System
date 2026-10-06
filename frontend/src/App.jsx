import { useState } from "react";
import "./App.css";
import SpotlightCard from "./components/SpotlightCard";

function App() {
  const [text, setText] = useState("");
  const [image, setImage] = useState(null);
  const [video, setVideo] = useState(null);
  const [pdf, setPdf] = useState(null);

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // ============================================================
  // ANALYZE CONTENT
  // ============================================================

  const handleAnalyze = async () => {
  // At least one supported input is required
  if (!text.trim() && !image && !pdf) {
    alert("Please enter some text, choose an image, or choose a PDF.");
    return;
  }

  // Video is not implemented yet
  if (video) {
    alert(
      "Video analysis will be available after the video processing module is implemented."
    );
    return;
  }

  // PDF is currently analyzed separately
  if (pdf && (text.trim() || image)) {
    alert(
      "Please analyze the PDF separately without adding text or an image."
    );
    return;
  }

  setLoading(true);
  setResult(null);

  const formData = new FormData();

  // Add text when available
  if (text.trim()) {
    formData.append("text", text);
  }

  // Add image when available
  if (image) {
    formData.append("image", image);
  }

  // Add PDF when available
  if (pdf) {
    formData.append("pdf", pdf);
  }

  try {
    const response = await fetch(
      "http://127.0.0.1:8000/predict",
      {
        method: "POST",
        body: formData,
      }
    );

    const data = await response.json();

    if (!response.ok || data.error) {
      throw new Error(
        data.error || "Prediction request failed."
      );
    }

    setResult(data);

  } catch (error) {
    console.error(error);

    alert(
      error.message ||
        "Could not connect to the AI backend."
    );
  } finally {
    setLoading(false);
  }
};

  // ============================================================
  // RESULT MODE
  // ============================================================

  const getModeName = (mode) => {
    if (mode === "text") {
      return "Text";
    }

    if (mode === "image") {
      return "Image";
    }

    if (mode === "multimodal") {
      return "Text + Image";
    }
    if (mode === "pdf") {
      return "PDF";
    }

    return mode;
  };

  // ============================================================
  // UI
  // ============================================================

  return (
    <div className="app">

      {/* ================= SPLASH CURSOR ================= */}


      {/* ================= NAVBAR ================= */}

      <nav className="navbar">

        <a href="#home" className="logo">
          <span className="logo-shield">🛡️</span>
          <span>AI Misinformation Detection System</span>
        </a>

        <div className="nav-links">
          <a href="#home">Home</a>
          <a href="#how-it-works">How It Works</a>
          <a href="#verify">Verify</a>
          <a href="#about">About</a>
        </div>

        <a href="#verify" className="nav-button">
          Start Verification
        </a>

      </nav>


      {/* ================= HERO ================= */}

      <section className="hero" id="home">

        <div className="badge">
          ✦ AI-POWERED MULTIMODAL VERIFICATION
        </div>

        <h1>
          Detect Misinformation
          <br />
          <span>Before You Share.</span>
        </h1>

        <p>
          Analyze news, images, videos, and documents using
          AI-powered multimodal verification.
        </p>

        <a href="#verify" className="hero-button">
          Start Verifying →
        </a>

      </section>


      {/* ================= VERIFY SECTION ================= */}

      <main className="main-container" id="verify">

        <div className="verification-card">

          <div className="card-header">

            <div>
              <span className="small-label">
                CONTENT ANALYSIS
              </span>

              <h2>Verify Your Content</h2>

              <p>
                Enter text or upload content for AI analysis.
              </p>
            </div>

            <div className="ai-status">
              <span className="status-dot"></span>
              AI System Ready
            </div>

          </div>


          {/* ================= TEXT ================= */}

          <div className="text-section">

            <label>
              <span>📝</span>
              News / Article Text
            </label>

            <textarea
              placeholder="Paste a headline, news article, social media post, or other text here..."
              value={text}
              onChange={(e) => setText(e.target.value)}
            />

            <div className="character-count">
              {text.length} characters
            </div>

          </div>


          {/* ================= UPLOADS ================= */}

          <div className="upload-grid">

            {/* ================= IMAGE ================= */}

            <SpotlightCard className="upload-card">

              <div className="upload-icon">
                📷
              </div>

              <h3>Image</h3>

              <p>
                Analyze photographs, screenshots, and news images.
              </p>

              <label className="file-button">
                Choose Image

                <input
                  type="file"
                  accept="image/*"
                  hidden
                  onChange={(e) => {
                    setImage(
                      e.target.files &&
                      e.target.files[0]
                        ? e.target.files[0]
                        : null
                    );

                    // Clear the input so the same file
                    // can be selected again if needed
                    e.target.value = "";
                  }}
                />

              </label>

              {image && (
                <div className="selected-file">
                  <span>
                    ✓ {image.name}
                  </span>

                  <button
                    type="button"
                    className="remove-file"
                    onClick={() => setImage(null)}
                    title="Remove image"
                  >
                    ✕
                  </button>
                </div>
              )}

            </SpotlightCard>


            {/* ================= VIDEO ================= */}

            <SpotlightCard className="upload-card">

              <div className="upload-icon">
                🎥
              </div>

              <h3>Video</h3>

              <p>
                Analyze video frames for potential manipulation.
              </p>

              <label className="file-button">
                Choose Video

                <input
                  type="file"
                  accept="video/*"
                  hidden
                  onChange={(e) => {
                    setVideo(
                      e.target.files &&
                      e.target.files[0]
                        ? e.target.files[0]
                        : null
                    );

                    e.target.value = "";
                  }}
                />

              </label>

              {video && (
                <div className="selected-file">
                  <span>
                    ✓ {video.name}
                  </span>

                  <button
                    type="button"
                    className="remove-file"
                    onClick={() => setVideo(null)}
                    title="Remove video"
                  >
                    ✕
                  </button>
                </div>
              )}

            </SpotlightCard>


            {/* ================= PDF ================= */}

            <SpotlightCard className="upload-card">

              <div className="upload-icon">
                📄
              </div>

              <h3>PDF Document</h3>

              <p>
                Extract and analyze text from PDF documents.
              </p>

              <label className="file-button">
                Choose PDF

                <input
                  type="file"
                  accept=".pdf,application/pdf"
                  hidden
                  onChange={(e) => {
                    setPdf(
                      e.target.files &&
                      e.target.files[0]
                        ? e.target.files[0]
                        : null
                    );

                    e.target.value = "";
                  }}
                />

              </label>

              {pdf && (
                <div className="selected-file">
                  <span>
                    ✓ {pdf.name}
                  </span>

                  <button
                    type="button"
                    className="remove-file"
                    onClick={() => setPdf(null)}
                    title="Remove PDF"
                  >
                    ✕
                  </button>
                </div>
              )}

            </SpotlightCard>

          </div>


          {/* ================= ANALYZE BUTTON ================= */}

          <button
            className="verify-button"
            onClick={handleAnalyze}
            disabled={loading}
          >

            <span>✦</span>

            {loading
              ? "Analyzing..."
              : "Analyze Content"}

            <span>→</span>

          </button>


          {/* ================= RESULT ================= */}

          {result && (
            <div className="result-card">

              <h2>Analysis Result</h2>

              {result.mode && (
                <p>
                  Analysis mode:{" "}
                  <strong>
                    {getModeName(result.mode)}
                  </strong>
                </p>
              )}

              <h3>
                {result.prediction}
              </h3>

              <p>
                Confidence: {result.confidence}%
              </p>

              <p>
                Fake probability:{" "}
                {result.fake_probability}%
              </p>

              <p>
                True probability:{" "}
                {result.true_probability}%
              </p>

            </div>
          )}


          <p className="privacy-note">
            🔒 Your content is processed securely.
          </p>

        </div>

      </main>


      {/* ================= HOW IT WORKS ================= */}

      <section
        className="how-section"
        id="how-it-works"
      >

        <div className="section-badge">
          HOW IT WORKS
        </div>

        <h2>
          One platform. Multiple signals.
        </h2>

        <p className="section-description">
          AI Misinformation Detector combines multiple types
          of information to provide a more comprehensive
          misinformation assessment.
        </p>


        <div className="steps">

          <div className="step">

            <div className="step-number">
              01
            </div>

            <div className="step-icon">
              📥
            </div>

            <h3>Provide Content</h3>

            <p>
              Enter news text or upload an image, video,
              or PDF document.
            </p>

          </div>


          <div className="step">

            <div className="step-number">
              02
            </div>

            <div className="step-icon">
              🧠
            </div>

            <h3>AI Analysis</h3>

            <p>
              Specialized AI models analyze the available
              content and extract meaningful features.
            </p>

          </div>


          <div className="step">

            <div className="step-number">
              03
            </div>

            <div className="step-icon">
              🔍
            </div>

            <h3>Multimodal Verification</h3>

            <p>
              Information from different modalities is
              combined to produce the final assessment.
            </p>

          </div>


          <div className="step">

            <div className="step-number">
              04
            </div>

            <div className="step-icon">
              📊
            </div>

            <h3>Understand the Result</h3>

            <p>
              View the prediction, confidence score,
              and supporting explanation.
            </p>

          </div>

        </div>

      </section>


      {/* ================= ABOUT ================= */}

      <section
        className="about-section"
        id="about"
      >

        <div className="section-badge">
          ABOUT THE SYSTEM
        </div>

        <h2>
          Built for multimodal misinformation detection
        </h2>

        <p>
          AI Misinformation Detection System is designed to
          analyze multiple forms of content rather than
          relying on text alone. Text, images, videos, and
          documents can provide different signals that
          contribute to the final prediction.
        </p>

      </section>


      {/* ================= FOOTER ================= */}

      <footer>

        <div className="footer-logo">
          🛡️ AI Misinformation Detection System
        </div>

        <p>
          AI-Powered Multimodal Misinformation Detection System
        </p>

        <span>
          © 2026 AI Misinformation Detection System
        </span>

      </footer>

    </div>
  );
}

export default App;