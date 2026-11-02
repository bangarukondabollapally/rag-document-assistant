import { useLayoutEffect, useRef, type ReactNode } from "react";
import styles from "./Sheet.module.css";

interface SheetProps {
  open: boolean;
  side: "left" | "right";
  label: string;
  onClose: () => void;
  children: ReactNode;
}

/** Native modal dialog, which provides focus trapping, Escape and a backdrop. */
export function Sheet({ open, side, label, onClose, children }: SheetProps) {
  const ref = useRef<HTMLDialogElement>(null);

  // A layout effect, so the dialog is open before children's effects scroll or focus.
  useLayoutEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);

  return (
    <dialog
      ref={ref}
      aria-label={label}
      className={`${styles.sheet} ${styles[side]}`}
      onClose={onClose}
      onClick={(event) => {
        if (event.target === ref.current) onClose();
      }}
    >
      <div className={styles.content}>{children}</div>
    </dialog>
  );
}
