import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, Image, FileWarning } from "lucide-react";
import { ACCEPTED_FORMATS } from "../utils/constants.js";

export default function ImageUploader({ onFileSelect, preview, file, disabled }) {
  const onDrop = useCallback(
    (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        onFileSelect(acceptedFiles[0]);
      }
    },
    [onFileSelect]
  );

  const { getRootProps, getInputProps, isDragActive, fileRejections } =
    useDropzone({
      onDrop,
      accept: ACCEPTED_FORMATS,
      maxFiles: 1,
      maxSize: 10 * 1024 * 1024, // 10 MB
      disabled,
    });

  return (
    <div>
      <div
        {...getRootProps()}
        className={`upload-zone${isDragActive ? " drag-active" : ""}`}
        id="upload-zone"
      >
        <input {...getInputProps()} id="file-input" />

        {!preview ? (
          <>
            <div className="upload-icon">
              <Upload size={28} />
            </div>
            <p className="upload-title">
              {isDragActive
                ? "Drop your MRI scan here..."
                : "Drag & drop an MRI scan here"}
            </p>
            <p className="upload-subtitle">
              or <strong>click to browse</strong> your files
            </p>
            <div className="upload-formats">
              <span className="format-tag">JPG</span>
              <span className="format-tag">PNG</span>
              <span className="format-tag">BMP</span>
              <span className="format-tag">TIFF</span>
            </div>
          </>
        ) : (
          <div className="image-preview-container">
            <img
              src={preview}
              alt="MRI Preview"
              className="image-preview"
              id="image-preview"
            />
            <div className="image-info">
              <Image size={16} />
              <span>{file?.name}</span>
              <span>·</span>
              <span>{(file?.size / 1024).toFixed(1)} KB</span>
            </div>
          </div>
        )}
      </div>

      {fileRejections.length > 0 && (
        <div className="error-box" id="upload-error">
          <FileWarning size={18} />
          <span>
            {fileRejections[0].errors[0]?.message ||
              "Invalid file. Please upload a valid image."}
          </span>
        </div>
      )}
    </div>
  );
}
