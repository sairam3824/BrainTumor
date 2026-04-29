/** Tumor class display configuration */
export const CLASS_CONFIG = {
  glioma: {
    label: "Glioma",
    color: "#ef4444",
    emoji: "🔴",
    description: "A tumor originating in the glial cells of the brain or spine.",
  },
  meningioma: {
    label: "Meningioma",
    color: "#f59e0b",
    emoji: "🟡",
    description: "A tumor arising from the meninges surrounding the brain.",
  },
  no_tumor: {
    label: "No Tumor",
    color: "#10b981",
    emoji: "🟢",
    description: "No tumor detected — the MRI scan appears normal.",
  },
  pituitary: {
    label: "Pituitary",
    color: "#8b5cf6",
    emoji: "🟣",
    description: "A tumor in the pituitary gland near the base of the brain.",
  },
};

/** Accepted image file extensions */
export const ACCEPTED_FORMATS = {
  "image/jpeg": [".jpg", ".jpeg"],
  "image/png": [".png"],
  "image/bmp": [".bmp"],
  "image/tiff": [".tif", ".tiff"],
};

/** Format a timestamp for display */
export function formatTimestamp(iso) {
  return new Date(iso).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}
