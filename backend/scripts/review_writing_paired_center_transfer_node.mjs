// SPDX-License-Identifier: Apache-2.0
// Independent Node.js replay of the saved paired-draft measurements.
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'

const sha = (bytes) => createHash('sha256').update(bytes).digest('hex')
const digest = /^[0-9a-f]{64}$/
const generators = new Set(['granite', 'olmo', 'qwen', 'smol2'])
const models = { first: 768, second: 512 }
const whitespace = /[\p{White_Space}\u001c-\u001f]+/gu
const normalized = (text) => text.replace(whitespace, ' ').replace(/^ | $/g, '')

function jsonFile(path) {
  const bytes = readFileSync(path)
  assert(bytes.length > 0 && bytes.length <= 2_097_152, 'JSON size is outside replay scope')
  return [JSON.parse(bytes), sha(bytes)]
}

function counts(items, key) {
  const totals = Object.create(null)
  for (const item of items) totals[key(item)] = (totals[key(item)] ?? 0) + 1
  return { ...totals }
}

function orderedSpans(spans) {
  assert.equal(spans.length, 3, 'Three exact ranges are required')
  let previous = 0
  for (const range of spans) {
    assert(range.length === 2 && range.every(Number.isSafeInteger), 'Invalid source range')
    const [start, end] = range
    assert(previous <= start && start < end, 'Source ranges overlap or are out of order')
    previous = end
  }
}

function fromHex(text) {
  const parts = typeof text === 'string' && /^0x([01])\.([\da-f]{1,13})p([+-]?\d+)$/.exec(text)
  assert(parts, 'Invalid saved binary floating point value')
  const [, whole, fraction, exponent] = parts
  const power = Number(exponent)
  assert(Number.isInteger(power) && -1074 <= power && power <= 0, 'Invalid saved binary exponent')
  return (Number(whole) + Number.parseInt(fraction, 16) / 16 ** fraction.length) * 2 ** power
}

function logOddsIncrease(before, after) {
  const bounded = (value) => {
    assert(Number.isFinite(value) && 0 <= value && value <= 1, 'Saved response is outside [0,1]')
    return Math.min(1 - 1e-7, Math.max(1e-7, value))
  }
  const a = bounded(before)
  const b = bounded(after)
  return Math.log((b * (1 - a)) / (a * (1 - b)))
}

function pairedMinimum(before, after, readings) {
  return Math.min(...Object.values(readings).map((model) =>
    logOddsIncrease(fromHex(model[before].score_hex), fromHex(model[after].score_hex))))
}

function scoreStudy(receipt) {
  assert.equal(receipt.schema, 'chipmunk.writing-paired-center-transfer/1')
  assert.deepEqual(receipt.forwards, { first: 63, second: 63 })
  assert.deepEqual(Object.keys(receipt.bindings).sort(), ['first', 'freeze', 'panels', 'second'])
  assert(Object.values(receipt.bindings).every((pin) => digest.test(pin)))
  const roster = {
    selection: { before: 9, edited: 9 },
    journal: { before: 21, edited: 21, qwen: 16, smol2: 16 },
    f1000: { self: 10, ordinary: 10, granite: 8, olmo: 6 },
  }
  assert.deepEqual(Object.keys(receipt.cases).sort(), Object.keys(roster).sort())
  const inputSets = {}
  for (const [panel, cases] of Object.entries(receipt.cases)) {
    assert.deepEqual(counts(cases, (row) => row.role), roster[panel], 'Unexpected panel roster')
    const canonicalBefore = new Map(cases.filter((row) => ['before', 'self'].includes(row.role))
      .map((row) => [row.group, row.pair[0]]))
    assert.equal(new Set(cases.map((row) => JSON.stringify([row.group, row.role]))).size, cases.length)
    inputSets[panel] = new Set()
    for (const row of cases) {
      const [after, before] = row.pair
      const unchanged = ['before', 'self'].includes(row.role)
      assert.deepEqual(before, canonicalBefore.get(row.group), 'Earlier version differs within a work')
      if (unchanged) assert.deepEqual(after, before)
      assert.equal(after.source_sha256 === before.source_sha256, unchanged)
      assert.deepEqual(after.neighbors, before.neighbors, 'Recorded flanks changed')
      assert.equal(before.neighbors.length, 2)
      assert([row.existing_flagged, row.existing_useful, row.native_app_eligible].every((flag) => typeof flag === 'boolean'))
      assert(!row.existing_useful || row.existing_flagged)
      assert(!unchanged || !row.native_app_eligible)
      for (const side of row.pair) {
        orderedSpans(side.spans)
        assert([side.source_sha256, side.center, side.context, ...side.neighbors].every((pin) => digest.test(pin)))
        inputSets[panel].add(side.context).add(side.center)
      }
    }
  }
  const held = new Set([...inputSets.journal, ...inputSets.f1000])
  assert(![...inputSets.selection].some((pin) => held.has(pin)), 'Calibration and held inputs overlap')
  const allInputs = new Set([...inputSets.selection, ...held])
  assert.equal(allInputs.size, 252)
  assert.deepEqual(Object.keys(receipt.measurements).sort(), Object.keys(models))
  for (const [model, observations] of Object.entries(receipt.measurements)) {
    assert.deepEqual(new Set(Object.keys(observations)), allInputs, 'Incomplete response roster')
    for (const { tokens, score_hex } of Object.values(observations)) {
      assert(Number.isSafeInteger(tokens) && 0 < tokens && tokens <= models[model])
      const score = fromHex(score_hex)
      assert(Number.isFinite(score) && 0 <= score && score <= 1)
    }
  }

  const scored = Object.fromEntries(Object.entries(receipt.cases).map(([panel, cases]) => [panel, cases.map((row) => {
    const [after, before] = row.pair
    const context = pairedMinimum(before.context, after.context, receipt.measurements)
    const center = pairedMinimum(before.center, after.center, receipt.measurements)
    return { ...row, scores: { context, calibrated_context: context, center, joint: Math.min(context, center) } }
  })]))
  const points = { context: Math.log(2) }
  for (const arm of ['calibrated_context', 'center', 'joint']) {
    const maximumByWork = new Map()
    for (const row of scored.selection) {
      const previous = maximumByWork.get(row.group) ?? -Infinity
      maximumByWork.set(row.group, Math.max(previous, row.scores[arm]))
    }
    points[arm] = [...maximumByWork.values()].sort((a, b) => b - a)[1]
    assert(Math.abs(receipt.result.points[arm] - points[arm]) <= 4e-15, `Published ${arm} point disagrees`)
  }
  assert(Math.abs(receipt.result.points.context - points.context) <= 4e-15, 'Published incumbent point disagrees')
  const summary = {}
  for (const panel of ['journal', 'f1000']) {
    summary[panel] = {}
    for (const [arm, point] of Object.entries(points)) {
      const generated = []
      const ordinary = []
      for (const row of scored[panel]) {
        const changed = !['before', 'self'].includes(row.role)
        const incumbent = changed && row.native_app_eligible && row.scores.context >= points.context
        const flagged = changed && (arm === 'context' ? row.scores[arm] >= point : row.scores[arm] > point)
        if (generators.has(row.role)) generated.push({ ...row, incumbent: incumbent || row.existing_useful, flagged })
        else ordinary.push({ ...row, incumbent: incumbent || row.existing_flagged, flagged })
      }
      const additions = generated.filter((row) => row.flagged && !row.incumbent)
      const baseline = new Set(ordinary.filter((row) => row.incumbent).map((row) => row.group))
      const alerts = new Set(ordinary.filter((row) => row.flagged).map((row) => row.group))
      summary[panel][arm] = {
        known: generated.length, existing: generated.filter((row) => row.incumbent).length,
        candidate: generated.filter((row) => row.flagged).length, added: additions.length,
        app_eligible_added: additions.filter((row) => row.native_app_eligible).length,
        added_works: new Set(additions.map((row) => row.group)).size,
        families: [...new Set(additions.map((row) => row.role))].sort(),
        genuine: ordinary.length, flagged: ordinary.filter((row) => row.flagged).length,
        false: alerts.size, prior_false: baseline.size, added_false: [...alerts].filter((key) => !baseline.has(key)).length,
      }
    }
  }
  assert.deepEqual(summary, receipt.result.summary, 'Held outcomes do not match')
  return { points, summary, pooled_responses: allInputs.size * Object.keys(models).length }
}

function scoreBatches(receipt, points) {
  const batches = receipt.boundary_batches
  assert.deepEqual(counts(batches, (batch) => batch.panel), { selection: 2, journal: 15, f1000: 5 })
  assert.equal(new Set(batches.map((row) => `${row.panel}/${row.index}`)).size, 22)
  const outcome = { generated: 0, generated_context: 0, generated_joint: 0, ordinary: 0, ordinary_context: 0, ordinary_joint: 0 }
  for (const row of batches) {
    const item = receipt.cases[row.panel][row.index]
    const [after, before] = item.pair
    assert(!item.native_app_eligible && row.change_ids.length > 0)
    assert.deepEqual(row.pins, [before.context, after.context, before.center, after.center], 'Batch input order differs')
    const [left, target, right] = row.paragraphs
    assert(row.paragraphs.length === 3 && row.paragraphs.every(Number.isSafeInteger)
      && left >= 0 && left <= target && target <= right && right === left + 1, 'Paragraph range differs')
    assert.deepEqual(Object.keys(row.readings).sort(), Object.keys(models))
    const context = []
    const center = []
    for (const [model, native] of Object.entries(row.readings)) {
      assert.equal(native.responses_hex.length, 4)
      assert.deepEqual(native.tokens, row.pins.map((pin) => receipt.measurements[model][pin].tokens))
      const [beforeContext, afterContext, beforeCenter, afterCenter] = native.responses_hex.map(fromHex)
      context.push(logOddsIncrease(beforeContext, afterContext))
      center.push(logOddsIncrease(beforeCenter, afterCenter))
    }
    const contextual = Math.min(...context) > points.calibrated_context
    const joint = Math.min(...context, ...center) > points.joint
    assert.equal(joint, row.joint, 'Native joint decision differs')
    assert.equal(contextual, row.calibrated_context, 'Native context decision differs')
    if (row.panel === 'selection') continue
    const key = generators.has(item.role) ? 'generated' : 'ordinary'
    outcome[key]++
    outcome[`${key}_context`] += Number(contextual)
    outcome[`${key}_joint`] += Number(joint)
  }
  assert.deepEqual(outcome, receipt.boundary_result, 'Native batch totals differ')
  return outcome
}

function loadDraft(path, version) {
  const bytes = readFileSync(path)
  assert(bytes.length > 0 && bytes.length <= 1_048_576, 'Draft exceeds size limit')
  assert.equal(sha(bytes), version.sha256, 'Original draft bytes differ')
  assert.equal(bytes.length, version.bytes)
  const signature = bytes.subarray(0, 2).toString('hex')
  const encoding = { fffe: 'utf-16le', feff: 'utf-16be' }[signature] ?? 'utf-8'
  const text = new TextDecoder(encoding, { fatal: true }).decode(bytes)
  assert(!text.includes('\0') && !text.includes('\uFFFD'))
  const codepoints = Array.from(text)
  assert.equal(codepoints.length, version.decoded_characters)
  assert.equal(sha(text), version.decoded_text_sha256, 'Decoded draft differs')
  assert.equal(version.mime_part, null)
  return { codepoints, pin: sha(text) }
}

function replaySource(path, focus, source) {
  const { codepoints, pin } = loadDraft(path, source)
  const slice = ([start, end]) => codepoints.slice(start, end).join('')
  const { context_spans: spans } = focus
  orderedSpans(spans)
  assert(spans[2][1] <= codepoints.length, 'Source range exceeds draft')
  assert.deepEqual([focus.start, focus.end], spans[1], 'Focus does not match middle source range')
  assert.equal(slice(spans[1]), focus.text, 'Exact changed sentence differs')
  const pieces = spans.map((span) => normalized(slice(span)))
  const leftGap = slice([spans[0][1], spans[1][0]])
  const rightGap = slice([spans[1][1], spans[2][0]])
  const gaps = [leftGap, rightGap]
  assert(gaps.every((gap) => normalized(gap) === ''), 'Non-whitespace separates the source ranges')
  const breaks = gaps.map((gap) => [...gap.matchAll(/\r?\n[ \t]*\r?\n/g)].length)
  assert.equal(breaks[0] + breaks[1], 1, 'Expected one preserved paragraph break')
  assert.equal(sha(pieces.join(' ')), focus.input_sha256, 'Context input differs')
  assert.equal(sha(pieces[1]), focus.center_input_sha256, 'Exact sentence input differs')
  return { spans, pin, breaks, neighbors: [sha(pieces[0]), sha(pieces[2])], slice }
}

function replayApp(study, app, folder, point) {
  assert.equal(app.cases.length, 4)
  assert.equal(app.simulated_paid_jobs, 4)
  assert.equal(app.unpaid_jobs, 0)
  const outcomes = []
  for (const item of app.cases) {
    assert(['journal', 'f1000'].includes(item.panel) && /^[a-z0-9-]+$/.test(item.group)
      && /^[a-z0-9]+$/.test(item.role), 'Invalid app case identifier')
    const index = study.cases[item.panel].findIndex((row) => row.group === item.group && row.role === item.role)
    assert(index >= 0, 'App case has no corresponding study case')
    const [recordedLater, recordedEarlier] = study.cases[item.panel][index].pair
    assert.deepEqual([item.earlier, item.later], [recordedEarlier, recordedLater], 'App receipt source commitment differs')
    const batch = study.boundary_batches.find((row) => row.panel === item.panel && row.index === index)
    assert(batch && item.status === (batch.joint ? 'joint_increase' : 'no_joint_increase'))
    if (folder) {
      const directory = join(folder, `${item.panel}-${item.group}-${item.role}`)
      const [data] = jsonFile(join(directory, 'comparison.json'))
      const [old] = jsonFile(join(directory, 'old.json'))
      const { paired_check: check, ...plain } = data
      const { paired_check: historical, ...oldPlain } = old
      assert.equal(data.format, 'plain_text')
      assert.equal(data.coordinate_space, 'decoded_source_unicode_codepoint')
      assert.equal(check.schema, 'chipmunk.paired-writing-review/3')
      assert.equal(historical.schema, 'chipmunk.paired-writing-review/2')
      assert.equal(historical.status, 'outside_scope')
      assert.deepEqual(oldPlain, plain, 'Earlier opt-out comparison differs')
      assert.equal(check.authorship_confidence_contribution, 0)
      assert.deepEqual(check.change_ids, batch.change_ids)
      const source = {}
      for (const [side, filename] of Object.entries({ earlier: 'earlier.txt', later: 'analyzed.txt' })) {
        const verified = replaySource(join(directory, filename), check[side], data[side])
        source[side] = verified
        assert.equal(verified.pin, item[side].source_sha256)
        assert.deepEqual(verified.spans, item[side].spans)
        assert.deepEqual(verified.neighbors, item[side].neighbors)
      }
      assert.deepEqual(source.earlier.breaks, source.later.breaks, 'Break placement differs between drafts')
      assert.deepEqual(data.changes.map((change) => change.id), check.change_ids)
      for (const change of data.changes) {
        assert.equal(change.role, 'body')
        for (const side of ['earlier', 'later']) {
          const { start, end, text, changed_tokens: tokens } = change[side]
          assert(check[side].start <= start && start <= end && end <= check[side].end)
          assert.equal(source[side].slice([start, end]), text)
          assert(tokens.every(([a, b]) => start <= a && a < b && b <= end))
        }
      }
      assert(check.responses.length === 2 && check.center_responses.length === 2)
      const response = Math.min(...[...check.responses, ...check.center_responses]
        .map((row) => logOddsIncrease(row.earlier, row.later)))
      assert.equal(check.status, response > point ? 'joint_increase' : 'no_joint_increase', 'Export direction differs')
      assert.equal(check.status, item.status, 'App JSON and published batch disagree')
    }
    outcomes.push(item.status)
  }
  return { linked_cases: outcomes.length, exact_drafts_checked: folder ? outcomes.length * 2 : 0,
    joint_increases: outcomes.filter((status) => status === 'joint_increase').length }
}

const [studyPath, appPath, folder] = process.argv.slice(2)
if (!studyPath || !appPath || process.argv.length > 5) {
  console.error('Usage: node review_writing_paired_center_transfer_node.mjs STUDY_JSON APP_RECEIPT_JSON [LOCAL_PAID_REPLAY_FOLDER]')
  process.exitCode = 2
} else {
  try {
    const [study, studyHash] = jsonFile(studyPath)
    const [app, appHash] = jsonFile(appPath)
    const result = scoreStudy(study)
    const boundary = scoreBatches(study, result.points)
    const appCases = replayApp(study, app, folder, result.points.joint)
    console.log(JSON.stringify({
      schema: 'chipmunk.writing-paired-center-node-replay/1', node: process.versions.node,
      v8: process.versions.v8, platform: process.platform, architecture: process.arch,
      study_sha256: studyHash, app_receipt_sha256: appHash,
      case_views: Object.values(study.cases).reduce((n, rows) => n + rows.length, 0),
      native_batches: study.boundary_batches.length, ...result, boundary_result: boundary, app: appCases,
    }, null, 2))
  } catch (error) {
    console.error(`Paired writing replay failed. ${error.message.split('\n')[0]}`)
    process.exitCode = 1
  }
}
