/**
 * Browser pre-hash (UX §5.1; BUILD_SPEC C-2): the SHA-256 of a file, computed locally before upload and sent as
 * `client_sha256`. The server compares it with its own hash and refuses the upload on a difference.
 *
 * SubtleCrypto has no incremental digest, so this is a FIPS 180-4 SHA-256 over 4 MiB slices: a large video is never held
 * in memory whole.
 */

const K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
])

export class Sha256 {
  private readonly h = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19])
  private readonly w = new Uint32Array(64)
  private readonly buf = new Uint8Array(64)
  private bufLen = 0
  private bytes = 0
  private done = false

  update(data: Uint8Array): this {
    if (this.done) throw new Error('Sha256: update after digest')
    this.bytes += data.length
    let i = 0
    if (this.bufLen > 0) {
      i = Math.min(64 - this.bufLen, data.length)
      this.buf.set(data.subarray(0, i), this.bufLen)
      this.bufLen += i
      if (this.bufLen < 64) return this
      this.block(this.buf, 0)
      this.bufLen = 0
    }
    for (; i + 64 <= data.length; i += 64) this.block(data, i)
    if (i < data.length) {
      this.buf.set(data.subarray(i), 0)
      this.bufLen = data.length - i
    }
    return this
  }

  /** Lower-case hex digest; the instance cannot be updated afterwards. */
  digestHex(): string {
    const bits = this.bytes * 8
    const pad = new Uint8Array((this.bufLen < 56 ? 56 : 120) - this.bufLen + 8)
    pad[0] = 0x80
    const view = new DataView(pad.buffer)
    view.setUint32(pad.length - 8, Math.floor(bits / 0x100000000))
    view.setUint32(pad.length - 4, bits % 0x100000000)
    this.update(pad)
    this.done = true
    return Array.from(this.h, (x) => x.toString(16).padStart(8, '0')).join('')
  }

  private block(p: Uint8Array, off: number): void {
    const w = this.w
    const h = this.h
    for (let t = 0; t < 16; t++) {
      const j = off + 4 * t
      w[t] = (p[j] << 24) | (p[j + 1] << 16) | (p[j + 2] << 8) | p[j + 3]
    }
    for (let t = 16; t < 64; t++) {
      const x = w[t - 2]
      const y = w[t - 15]
      const s1 = ((x >>> 17) | (x << 15)) ^ ((x >>> 19) | (x << 13)) ^ (x >>> 10)
      const s0 = ((y >>> 7) | (y << 25)) ^ ((y >>> 18) | (y << 14)) ^ (y >>> 3)
      w[t] = (s1 + w[t - 7] + s0 + w[t - 16]) | 0
    }
    let a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7]
    for (let t = 0; t < 64; t++) {
      const sigmaE = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7))
      const t1 = (hh + sigmaE + ((e & f) ^ (~e & g)) + K[t] + w[t]) | 0
      const sigmaA = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10))
      const t2 = (sigmaA + ((a & b) ^ (a & c) ^ (b & c))) | 0
      hh = g; g = f; f = e; e = (d + t1) | 0; d = c; c = b; b = a; a = (t1 + t2) | 0
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh
  }
}

export const HASH_CHUNK_BYTES = 4 << 20

/** SHA-256 of a Blob (or File), read slice by slice. */
export async function sha256Blob(blob: Blob, chunkBytes = HASH_CHUNK_BYTES): Promise<string> {
  const hash = new Sha256()
  for (let off = 0; off < blob.size; off += chunkBytes) {
    hash.update(new Uint8Array(await blob.slice(off, off + chunkBytes).arrayBuffer()))
  }
  return hash.digestHex()
}
