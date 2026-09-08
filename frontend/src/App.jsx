import { useState } from "react";
import "./App.css";

function App() {
  const [text, setText] = useState("");
  const [image, setImage] = useState(null);
  const [video, setVideo] = useState(null);
  const [pdf, setPdf] = useState(null);

  return (
    <div className="app">

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


          {/* TEXT */}
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


          {/* UPLOADS */}
          <div className="upload-grid">

            {/* IMAGE */}
            <div className="upload-card">

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
                  onChange={(e) =>
                    setImage(e.target.files[0])
                  }
                />
              </label>

              {image && (
                <div className="selected-file">
                  ✓ {image.name}
                </div>
              )}

            </div>


            {/* VIDEO */}
            <div className="upload-card">

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
                  onChange={(e) =>
                    setVideo(e.target.files[0])
                  }
                />
              </label>

              {video && (
                <div className="selected-file">
                  ✓ {video.name}
                </div>
              )}

            </div>


            {/* PDF */}
            <div className="upload-card">

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
                  onChange={(e) =>
                    setPdf(e.target.files[0])
                  }
                />
              </label>

              {pdf && (
                <div className="selected-file">
                  ✓ {pdf.name}
                </div>
              )}

            </div>

          </div>


          {/* ANALYZE BUTTON */}
          <button className="verify-button">
            <span>✦</span>
            Analyze Content
            <span>→</span>
          </button>

          <p className="privacy-note">
            🔒 Your content is processed securely.
          </p>

        </div>

      </main>


      {/* ================= HOW IT WORKS ================= */}
      <section className="how-section" id="how-it-works">

        <div className="section-badge">
          HOW IT WORKS
        </div>

        <h2>
          One platform. Multiple signals.
        </h2>

        <p className="section-description">
         AI Misinformation Detector combines multiple types of information to
          provide a more comprehensive misinformation assessment.
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
      <section className="about-section" id="about">

        <div className="section-badge">
          ABOUT THE SYSTEM
        </div>

        <h2>
          Built for multimodal misinformation detection
        </h2>

        <p>
          AI Misinformation Detection System is designed to analyze multiple forms of
          content rather than relying on text alone. Text,
          images, videos, and documents can provide different
          signals that contribute to the final prediction.
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