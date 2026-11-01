/** Chipmunk Robotics' circular mark, shared by the workspace and sign-in screen. */
export default function BrandMark({ className = '' }: { className?: string }) {
  return (
    <svg viewBox="0 0 44 44" aria-hidden="true" className={className} fill="none">
      <path d="M33 10A16 16 0 1 0 33 34" stroke="currentColor" strokeWidth="6" />
      <path d="M29 17a8 8 0 1 0 0 10" stroke="currentColor" strokeWidth="3" />
      <circle cx="35" cy="22" r="3" fill="#b75d3c" />
    </svg>
  )
}
