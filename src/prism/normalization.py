"""Deterministic date, quantity and short-answer normalization."""
import re

MONTHS = ('January', 'February', 'March', 'April', 'May', 'June', 'July',
          'August', 'September', 'October', 'November', 'December')

def humanize_dates(text):
    'Rewrite ISO and compact dates as spoken dates.'
    import re as _re

    def iso(m):
        y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
        if not 1 <= mo <= 12 or not 1 <= d <= 31:
            return m.group(0)
        return '%d %s %s' % (d, MONTHS[mo - 1], y)

    def compact(m):
        y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
        if not 1 <= mo <= 12 or not 1 <= d <= 31:
            return m.group(0)
        return '%d %s %s' % (d, MONTHS[mo - 1], y)

    text = _re.sub(r'\b((?:19|20)\d{2})-(\d{2})-(\d{2})\b', iso, text)
    text = _re.sub(r'\b((?:19|20)\d{2})(\d{2})(\d{2})\b', compact, text)
    return text

REL_ANCHOR = __import__('re').compile(
    r'\(\s*(?:as of|on)\s+((?:19|20)\d{2})-(\d{2})-(\d{2})\s*\)', 2 | 16)

REL_UNIT = __import__('re').compile(
    r'\b(this|last|next|the)\s+'
    r'(month|week|weekend|year|day|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b', 2 | 16)

def _shift_month(year, month, delta):
    zero = month - 1 + delta
    return year + zero // 12, zero % 12 + 1

def resolve_relative_dates(text):
    'Turn "last week (as of 2023-07-06)" into the calendar period it names.'
    import datetime as _dt
    import re as _re

    match = REL_ANCHOR.search(text)
    if not match:
        return text
    year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
    try:
        _dt.date(year, month, day)
    except ValueError:
        return text
    label = '%d %s %d' % (day, MONTHS[month - 1], year)
    rel = REL_UNIT.search(text)
    if not rel:
        return text
    tense, unit = rel.group(1).lower(), rel.group(2).lower()
    if unit == 'month':
        y2, m2 = _shift_month(year, month, {'this': 0, 'last': -1, 'next': 1}.get(tense, 0))
        replacement = '%s %d' % (MONTHS[m2 - 1], y2)
    elif unit == 'week':
        replacement = 'the week of ' + label if tense == 'this' else (
            'the week before ' + label if tense == 'last' else 'the week after ' + label)
    elif unit == 'weekend':
        replacement = 'the weekend of ' + label if tense == 'this' else (
            'the weekend before ' + label if tense == 'last' else 'the weekend after ' + label)
    elif unit == 'year':
        replacement = str(year) if tense == 'this' else (
            str(year - 1) if tense == 'last' else str(year + 1))
    elif unit == 'day':
        replacement = label
    else:
        replacement = ('the %s before %s' % (unit.capitalize(), label) if tense == 'last'
                       else 'the %s after %s' % (unit.capitalize(), label) if tense == 'next'
                       else unit.capitalize())
    out = text[:rel.start()] + replacement + text[rel.end():]
    out = REL_ANCHOR.sub('', out)
    return _re.sub(r'\s{2,}', ' ', out).strip(' ,;') or text

def _recorded_date(value):
    "Parse a turn timestamp like '1:56 pm on 8 May, 2023'."
    import re as _re
    m = _re.search(r'(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})', value or '')
    if not m:
        return None
    name = m.group(2)[:3].title()
    if name not in [x[:3] for x in MONTHS]:
        return None
    return int(m.group(3)), [x[:3] for x in MONTHS].index(name) + 1, int(m.group(1))

def present_answer(text, anchor=None):
    'Normalise dates; resolve a relative phrase against the cited evidence.'
    import re as _re
    if anchor is not None:


        m = _re.search(r'(\d{4})-(\d{2})-(\d{2})', anchor)
        if m:
            y, mo, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        else:
            parsed = _recorded_date(anchor)
            if parsed is None:
                return humanize_dates(resolve_relative_dates(text))
            y, mo, day = parsed
        label = '%d %s %d' % (day, MONTHS[mo - 1], y)
        _fix_rel = True
        if _fix_rel:
            import datetime as _dt
            _ytt = _re.search(r'\b(yesterday|today|tomorrow)(?:\s+(?:morning|afternoon|evening|night))?\b', text, _re.I)
            if _ytt and not _re.search(r'\b\d{4}\b', text):
                try:
                    _d0 = _dt.date(y, mo, day) + _dt.timedelta(
                        days={'yesterday': -1, 'today': 0, 'tomorrow': 1}[_ytt.group(1).lower()]
                    )
                    _dlbl = '%d %s %d' % (_d0.day, MONTHS[_d0.month - 1], _d0.year)
                    text = text[:_ytt.start()] + _dlbl + text[_ytt.end():]
                except ValueError:
                    pass
        rel = REL_UNIT.search(text)
        if rel and not REL_ANCHOR.search(text):
            tense, unit = rel.group(1).lower(), rel.group(2).lower()
            if _fix_rel and _re.match(r'\s*(?:before|after|of)\s+(?:\d{1,2}\s+[A-Za-z]+|[A-Za-z]+\s+\d{1,2}|\d{4})', text[rel.end():], _re.I):


                if tense in ('last', 'next'):
                    text = text[:rel.start()] + 'the ' + rel.group(2) + text[rel.end():]
                return humanize_dates(resolve_relative_dates(text))
            if _fix_rel and tense == 'the':
                return humanize_dates(resolve_relative_dates(text))
            if unit == 'month':
                y2, m2 = _shift_month(y, mo, {'this': 0, 'last': -1, 'next': 1}.get(tense, 0))
                repl = '%s %d' % (MONTHS[m2 - 1], y2)
            elif unit == 'year':
                repl = str(y) if tense == 'this' else (str(y - 1) if tense == 'last' else str(y + 1))
            elif unit == 'week':
                repl = ('the week of ' + label) if tense == 'this' else (
                    'the week before ' + label if tense == 'last' else 'the week after ' + label)
            elif unit == 'weekend':
                repl = ('the weekend of ' + label) if tense == 'this' else (
                    'the weekend before ' + label if tense == 'last' else 'the weekend after ' + label)
            elif unit == 'day':
                repl = label
            else:
                repl = ('the %s before %s' % (unit.capitalize(), label) if tense == 'last'
                        else 'the %s after %s' % (unit.capitalize(), label) if tense == 'next'
                        else unit.capitalize())
            if not (_fix_rel and repl.lower() in text.lower()):
                text = text[:rel.start()] + repl + text[rel.end():]
    out = humanize_dates(resolve_relative_dates(text))
    if True:
        out = _re.sub(
            r'the (week|weekend) after (\d{1,2} [A-Za-z]+ \d{4}) before (\d{1,2} [A-Za-z]+ \d{4})',
            lambda m: 'the %s before %s' % (m.group(1), m.group(2)),
            out,
            flags=_re.I
        )
    return out

COUNT_QUESTION = re.compile(r'^\s*how (many|much)\b', re.I)

NUMBER_WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
                'eight', 'nine', 'ten', 'eleven', 'twelve']

BARE = re.compile(r'^\s*(\d{1,2})\s*(.*)$')

def canonical_count(question, text):
    if not COUNT_QUESTION.match(str(question or '')):
        return text
    match = BARE.match(str(text or ''))
    if not match:
        return text
    value = int(match.group(1))
    if value >= len(NUMBER_WORDS):
        return text
    rest = match.group(2).strip(' .,;')
    return NUMBER_WORDS[value] + ((' ' + rest) if rest else '')
YESNO = re.compile(r'^\s*(did|does|do|is|are|was|were|has|have|had|can|could|would|will|should)\b', re.I)

NEG = re.compile(r'^\s*(no\b|not\b|never\b|unlikely|probably not|likely not|cannot|can\'t|'
                 r'doesn\'t|didn\'t|isn\'t|wasn\'t|hasn\'t|haven\'t|won\'t|would not|'
                 r'wouldn\'t|no, )', re.I)

POS = re.compile(r'^\s*(yes\b|yeah|yep|definitely|absolutely|certainly|indeed|sure|'
                 r'likely yes|probably yes|of course)', re.I)

def canonical_verdict(question, answer):
    'Return the answer with its opening verdict normalised, or None if unchanged.'
    if not YESNO.match(str(question or '')):
        return None
    text = str(answer or '').strip()
    if not text:
        return None
    for pattern, word in ((NEG, 'No'), (POS, 'Yes')):
        m = pattern.match(text)
        if not m:
            continue
        rest = text[m.end():].lstrip(' ,;:-')
        rebuilt = word + (', ' + rest if rest else '')
        return rebuilt if rebuilt != text else None
    return None

HEDGE = re.compile(r'^\s*(approximately|about|around|roughly|probably|likely|maybe|perhaps)\s+(.+)$', re.I)

def strip_leading_hedge(answer):
    'Drop a leading hedge qualifier ("about five years" -> "five years").'
    m = HEDGE.match(str(answer or ''))
    if not m:
        return None
    rest = m.group(2).strip()
    if not rest or rest == str(answer).strip():
        return None
    return rest

TRAILER = re.compile(r'\s*\((as of|at the time|in|during)[^)]*\)\s*$', re.I)

NOTE_PREFIX = re.compile(r'^\s*(inference|answer|note)\s*[:\-]\s*(.+)$', re.I)

def strip_trailer(answer):
    'Drop a trailing parenthetical aside that the references never carry.'
    text = str(answer or '')
    cleaned = TRAILER.sub('', text).strip()
    return cleaned if cleaned and cleaned != text.strip() else None

NAME_PREFIX = None

def strip_leading_subject(answer, participants):
    'Drop a leading subject name ("John attended the convention" -> "attended ...").'
    text = str(answer or '').strip()
    if not text:
        return None
    low = text.lower()
    for name in (participants or []):
        for prefix in (name.lower() + ' ', name.lower() + "'s "):
            if low.startswith(prefix):
                rest = text[len(prefix):].strip()
                if len(rest.split()) >= 2:
                    return rest
    return None

INLINE_HEDGE = re.compile(r'\s*\b(probably|likely|approximately|possibly|perhaps|presumably|'
                          r'certainly|definitely|actually|basically|essentially)\b\s*', re.I)

def drop_inline_hedges(answer, max_words=12):
    'Remove hedge adverbs anywhere in a short answer.'
    text = str(answer or '').strip()
    if not text or len(text.split()) > max_words:
        return None
    cleaned = INLINE_HEDGE.sub(' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip(' ,;')
    if not cleaned or cleaned == text or not cleaned.split():
        return None
    return cleaned

ASOF = re.compile(r'(\s*\(\s*as of[^)]*\)|[,;]?\s*as of\s+[^;]*)\s*$', re.I)

def strip_as_of(answer):
    'Drop a trailing anchor echo such as "as of 20 July 2023".'
    text = str(answer or '').strip()
    cleaned = ASOF.sub('', text).strip(' ,;')
    if not cleaned or cleaned == text or len(cleaned.split()) < 2:
        return None
    return cleaned

def strip_note_prefix(answer):
    'Drop a leading label such as "Inference:" left by the reader.'
    m = NOTE_PREFIX.match(str(answer or ''))
    if not m:
        return None
    rest = m.group(2).strip()
    return rest if rest and rest != str(answer).strip() else None

DURATION_QUESTION = re.compile(
    r'\bhow long|\bhow many (days|weeks|months|years|hours)|\bhow much time\b', re.I)

HAS_UNIT = re.compile(r'\b(day|days|week|weeks|month|months|year|years|hour|hours|minute|minutes|decade|decades)\b', re.I)

def needs_unit(question, answer):
    if not DURATION_QUESTION.search(str(question or '')):
        return False
    return not HAS_UNIT.search(str(answer or ''))
