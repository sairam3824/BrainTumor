export default function AboutPage() {
  const pipeline = [
    {
      title: "Image Upload",
      desc: "User uploads a brain MRI scan through the drag-and-drop interface. The image is sent to the Flask API backend.",
    },
    {
      title: "Preprocessing",
      desc: "The image is converted to grayscale, resized to 128×128 pixels, and normalized to [0, 1] range for model input.",
    },
    {
      title: "Feature Extraction",
      desc: "A convolutional autoencoder encodes the image into a 128-dimensional bottleneck feature vector, capturing key patterns.",
    },
    {
      title: "Dimensionality Reduction",
      desc: "PCA reduces the 128 features to 50 principal components, removing noise and improving classification accuracy.",
    },
    {
      title: "Classification",
      desc: "An RBF-kernel SVM classifier predicts the tumor class with calibrated probability estimates for confidence scoring.",
    },
  ];

  const cards = [
    {
      icon: "🧠",
      bg: "rgba(139,92,246,0.1)",
      title: "Deep Learning Autoencoder",
      text: "A convolutional denoising autoencoder with 3 encoding layers, a 128-unit bottleneck, and 3 transpose-conv decoding layers. Trained on brain MRI images to learn compressed representations.",
    },
    {
      icon: "📊",
      bg: "rgba(6,182,212,0.1)",
      title: "PCA + SVM Classifier",
      text: "Principal Component Analysis reduces features to 50 dimensions. A Support Vector Machine with RBF kernel (C=300, γ=0.01) performs the final 4-class classification.",
    },
    {
      icon: "🏥",
      bg: "rgba(16,185,129,0.1)",
      title: "4 Tumor Classes",
      text: "Glioma, Meningioma, Pituitary tumors, and No Tumor. The model provides per-class probability scores alongside the top prediction.",
    },
    {
      icon: "⚡",
      bg: "rgba(245,158,11,0.1)",
      title: "Real-Time Inference",
      text: "Models are loaded once at server startup. Each prediction completes in under a second, providing instant feedback to clinicians.",
    },
  ];

  return (
    <div className="page-container" id="about-page">
      <header className="page-header">
        <h1 className="page-title">About NeuroScan AI</h1>
        <p className="page-subtitle">
          An AI-powered brain tumor classification system combining deep learning
          feature extraction with classical machine learning classification.
        </p>
      </header>

      {/* Feature Cards */}
      <div className="about-grid">
        {cards.map((c) => (
          <div key={c.title} className="about-card glass-card">
            <div
              className="about-card-icon"
              style={{ background: c.bg }}
            >
              {c.icon}
            </div>
            <h3 className="about-card-title">{c.title}</h3>
            <p className="about-card-text">{c.text}</p>
          </div>
        ))}
      </div>

      {/* ML Pipeline Steps */}
      <div style={{ maxWidth: 700, margin: "48px auto 0" }}>
        <h2 style={{ textAlign: "center", fontWeight: 700, marginBottom: 8 }}>
          How It Works
        </h2>
        <p
          style={{
            textAlign: "center",
            color: "#94a3b8",
            fontSize: "0.95rem",
            marginBottom: 32,
          }}
        >
          The full prediction pipeline from image upload to result.
        </p>

        <div className="pipeline-steps">
          {pipeline.map((step, i) => (
            <div key={i} className="pipeline-step glass-card" style={{ padding: 20 }}>
              <div className="step-number">{i + 1}</div>
              <div className="step-content">
                <h4>{step.title}</h4>
                <p>{step.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
