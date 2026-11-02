import type { ChangeEvent } from "react";
import styles from "./UploadButton.module.css";

interface UploadButtonProps {
  disabled?: boolean;
  onUpload: (file: File) => void;
}

export function UploadButton({ disabled = false, onUpload }: UploadButtonProps) {
  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    // Cleared so that choosing the same file again still fires a change.
    event.target.value = "";
    if (file) onUpload(file);
  }

  return (
    <label className={`${styles.button} ${disabled ? styles.disabled : ""}`}>
      <input
        type="file"
        className="sr-only"
        accept="application/pdf,.pdf"
        disabled={disabled}
        onChange={handleChange}
      />
      Upload PDF
    </label>
  );
}
