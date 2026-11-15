'use strict'

// SPDX-License-Identifier: MIT
// Recount the published scalar experiment without the Python model or verifier.

const assert = require('node:assert/strict')
const { readFileSync } = require('node:fs')

const arms = ['dense', 'both_fills', 'same_winner']
const levels = ['strict', 'wider']

function options(row) {
  if (row.broad === null) {
    assert.equal(row.response, null)
    return []
  }
  const [start, end] = row.broad
  assert.equal(end - start, 32000)
  assert.equal(start % 16000, 0)
  assert.ok(0 <= start && end <= 96000)
  assert.equal(row.response.windows.length, 3)
  return row.response.windows.flatMap((values, index) => {
    if (values === null) return []
    assert.deepEqual(Object.keys(values).sort(), ['dense', 'to_flanks', 'to_mean'])
    assert.ok(Object.values(values).every(Number.isFinite))
    const first = start + index * 8000
    return [{ ...values, index, bounds: [first, first + 16000], minimum: Math.min(values.to_mean, values.to_flanks) }]
  })
}

function winner(row, arm) {
  const choices = options(row)
  const ranked = field => choices.slice().sort((a, b) => b[field] - a[field] || a.index - b.index)
  if (!choices.length) return null
  if (arm === 'dense') {
    const first = ranked('dense')[0]
    return { score: first.dense, bounds: first.bounds }
  }
  let first = ranked('minimum')[0]
  if (arm === 'same_winner') {
    const means = ranked('to_mean')
    const flanks = ranked('to_flanks')
    if (means[0].index !== flanks[0].index || means[0].to_mean === means[1]?.to_mean
        || flanks[0].to_flanks === flanks[1]?.to_flanks) return null
    first = means[0]
  }
  return { score: first.minimum, bounds: first.bounds }
}

function useful(range, truth) {
  if (range === null) return false
  const [first, last] = range
  const [start, end] = truth
  const overlap = Math.max(0, Math.min(last, end) - Math.max(first, start))
  return overlap / (end - start) >= 0.5 && overlap / (last - first + end - start - overlap) >= 0.30
}

const receipt = JSON.parse(readFileSync(process.argv[2], 'utf8'))
assert.equal(receipt.schema, 'chipmunk.audio-finite-window-attribution/1')
const points = {}
for (const arm of arms) {
  const groups = new Map()
  for (const row of receipt.panels.development) {
    if (row.truth !== null) continue
    const selected = winner(row, arm)
    if (selected) groups.set(row.group, Math.max(groups.get(row.group) ?? -Infinity, selected.score))
  }
  const ordered = Array.from(groups.values()).sort((a, b) => b - a)
  const floor = arm === 'dense' ? -Infinity : 0
  points[arm] = ordered.length < 2 && arm === 'same_winner'
    ? { ...points.both_fills, groups: groups.size, uses_both_fills_points: true }
    : { strict: Math.max(floor, ordered[0]), wider: Math.max(floor, ordered[1]), groups: groups.size }
}
assert.deepEqual(points, receipt.points)

let countsChecked = 0
for (const [name, rows] of Object.entries(receipt.panels)) {
  for (const arm of arms) {
    for (const level of levels) {
      const totals = new Map()
      const choices = rows.map(row => {
        const choice = winner(row, arm)
        const refined = choice !== null && choice.score > points[arm][level]
        return { row, refined, range: refined ? choice.bounds : row.broad }
      })
      const known = receipt.results[name][arm][level]
      for (const { row, refined, range } of choices) {
        const values = totals.get(row.condition) ?? {}
        totals.set(row.condition, values)
        const add = (key, number = 1) => { values[key] = (values[key] ?? 0) + Number(number) }
        if (row.truth) {
          const prior = useful(row.broad, row.truth)
          const after = useful(range, row.truth)
          add('edits')
          add('alerted_edits', row.broad !== null)
          add('useful_before', prior)
          add('useful_after', after)
          add('lost_useful', prior && !after)
          add('added_useful', after && !prior)
          add('misplaced_refinements', refined && !after)
          if (after && !prior) {
            const sameClock = choices.some(other => other.row.truth === null && other.row.group === row.group
              && other.row.condition === row.condition && useful(other.range, row.truth))
            add('added_useful_without_same_clock_control', !sameClock)
          }
        } else {
          const kind = ['background', 'background_only', 'isolated'].includes(row.kind)
            ? 'isolated' : row.kind === 'resyn' ? 'same_words' : 'ordinary'
          add(kind)
          add(`${kind}_alerts`, row.broad !== null)
          add(`${kind}_refined`, refined)
        }
      }
      assert.deepEqual(Object.fromEntries(totals), known.by_condition)
      countsChecked += totals.size
    }
  }
}
process.stdout.write(`${countsChecked} condition totals and all six operating points reproduced.\n`)
