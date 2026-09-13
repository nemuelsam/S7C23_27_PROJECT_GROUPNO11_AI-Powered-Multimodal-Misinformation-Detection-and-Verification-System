import { useRef } from "react";
import "./SpotlightCard.css";

function SpotlightCard({
  children,
  className = "",
  spotlightColor = "rgba(37, 99, 235, 0.12)",
}) {
  const cardRef = useRef(null);
  const spotlightRef = useRef(null);

  const handleMouseMove = (e) => {
    if (!cardRef.current || !spotlightRef.current) return;

    const rect = cardRef.current.getBoundingClientRect();

    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    spotlightRef.current.style.left = `${x}px`;
    spotlightRef.current.style.top = `${y}px`;
  };

  return (
    <div
      ref={cardRef}
      className={`spotlight-card ${className}`}
      onMouseMove={handleMouseMove}
    >
      <div
        ref={spotlightRef}
        className="spotlight-effect"
        style={{
          background: `radial-gradient(
            circle,
            ${spotlightColor} 0%,
            transparent 65%
          )`,
        }}
      />

      <div className="spotlight-content">
        {children}
      </div>
    </div>
  );
}

export default SpotlightCard;