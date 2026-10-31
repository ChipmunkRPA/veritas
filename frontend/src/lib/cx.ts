/** Tiny class joiner; avoids a dependency. */
export const cx = (...parts: Array<string | false | null | undefined>) => parts.filter(Boolean).join(' ')
