import { useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { useMediaQuery } from "../state/useMediaQuery";
import { SendIcon, StopIcon } from "./Icons";
import styles from "./QueryInput.module.css";

const MAX_LENGTH = 2000;
const MAX_HEIGHT_PX = 180;

interface QueryInputProps {
  busy: boolean;
  blockedReason: string | null;
  conversationMemory: boolean | null;
  onSubmit: (question: string) => void;
  onStop: () => void;
}

export function QueryInput({ busy, blockedReason, conversationMemory, onSubmit, onStop }: QueryInputProps) {
  const [value, setValue] = useState("");
  const textarea = useRef<HTMLTextAreaElement>(null);
  // A phone keyboard's Enter key should add a line; the button sends.
  const touch = useMediaQuery("(pointer: coarse)");
  const blocked = blockedReason !== null;
  const canSend = !blocked && !busy && value.trim().length > 0;

  function resize(element: HTMLTextAreaElement) {
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, MAX_HEIGHT_PX)}px`;
  }

  function send() {
    if (!canSend) return;
    onSubmit(value.trim());
    setValue("");
    if (textarea.current) textarea.current.style.height = "auto";
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    send();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !touch && !event.nativeEvent.isComposing) {
      event.preventDefault();
      send();
    }
  }

  return (
    <form className={styles.form} onSubmit={handleSubmit}>
      <div className={styles.inner}>
        <div className={styles.box}>
          <label htmlFor="question" className="sr-only">
            Question about the document
          </label>
          <textarea
            id="question"
            ref={textarea}
            className={styles.textarea}
            rows={1}
            maxLength={MAX_LENGTH}
            value={value}
            disabled={blocked}
            placeholder={blockedReason ?? "Ask a question about this document"}
            onChange={(event) => {
              setValue(event.target.value);
              resize(event.target);
            }}
            onKeyDown={handleKeyDown}
          />
          {busy ? (
            <button type="button" className={styles.button} aria-label="Stop generating" onClick={onStop}>
              <StopIcon />
            </button>
          ) : (
            <button type="submit" className={`${styles.button} ${styles.send}`} aria-label="Send question" disabled={!canSend}>
              <SendIcon />
            </button>
          )}
        </div>
        {conversationMemory === false && (
          <p className={styles.hint}>Each question is answered on its own. Follow-ups can't see earlier answers.</p>
        )}
      </div>
    </form>
  );
}
