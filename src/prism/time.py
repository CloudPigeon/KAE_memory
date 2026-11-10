'Deterministic temporal parsing and normalization.'
import calendar
from datetime import timedelta

from dateutil.parser import parse
from dateutil.relativedelta import relativedelta

_UNIT = {'day': 'days', 'days': 'days', 'week': 'weeks', 'weeks': 'weeks',
         'month': 'months', 'months': 'months', 'year': 'years', 'years': 'years'}

_COUNT = {'a': 1, 'an': 1, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
          'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
          'ten': 10, 'eleven': 11, 'twelve': 12}

_MONTH = {name.lower(): index for index, name in enumerate(calendar.month_name) if name}
_MONTH.update({name.lower(): index for index, name in enumerate(calendar.month_abbr) if name})

_WEEKDAY = {'mon': 0, 'monday': 0, 'tue': 1, 'tuesday': 1, 'wed': 2, 'wednesday': 2,
            'thu': 3, 'thursday': 3, 'fri': 4, 'friday': 4, 'sat': 5, 'saturday': 5,
            'sun': 6, 'sunday': 6}


_EXACT = {
    'today': (0, 'day'), 'now': (0, 'day'),
    'this day': (0, 'day'), 'tonight': (0, 'night'),
    'this morning': (0, 'day'), 'this afternoon': (0, 'day'), 'this evening': (0, 'night'),
    'tomorrow': (1, 'day'), 'tomorrow morning': (1, 'day'),
    'tomorrow afternoon': (1, 'day'), 'tomorrow evening': (1, 'night'),
    'tomorrow night': (1, 'night'),
    'yesterday': (-1, 'day'), 'yesterday morning': (-1, 'day'),
    'yesterday afternoon': (-1, 'day'), 'yesterday evening': (-1, 'night'),
    'yesterday night': (-1, 'night'), 'last night': (-1, 'night'),
    'the day before yesterday': (-2, 'day'), 'day before yesterday': (-2, 'day'),
    'last evening': (-1, 'night'),
}


_RANGES = {
    'last weekend': 'last_weekend', 'the weekend before': 'last_weekend',
    'last week': 'last_week', 'the week before': 'last_week',
    'this week': 'this_week', 'next week': 'next_week',
    'last month': 'last_month', 'the month before': 'last_month',
    'this month': 'this_month', 'next month': 'next_month',
    'last year': 'last_year', 'this year': 'this_year', 'next year': 'next_year',
}


_VAGUE = ('recently', 'a while ago', 'sometime', 'some time ago', 'lately',
          'the other day', 'not long ago', 'earlier', 'a few years ago')


_APPROXIMATE = ('a few', 'few', 'several', 'some', 'a handful of')


def _is_approximate(text):
    text = ' '.join(text.split())
    return any(text == word or text.startswith(word + ' ') for word in _APPROXIMATE)


class _Ordinal:
    'Observation-order stamp for releases without calendar time.'

    def __init__(self, value):
        self.value = int(value)

    def __ge__(self, other):
        return self.value >= other.value

    def __gt__(self, other):
        return self.value > other.value

    def __le__(self, other):
        return self.value <= other.value

    def __lt__(self, other):
        return self.value < other.value

    def __eq__(self, other):
        return isinstance(other, _Ordinal) and self.value == other.value

    def __hash__(self):
        return hash(("ordinal", self.value))

    def isoformat(self):
        return 'ordinal:%d' % self.value


def is_ordinal(text):
    return isinstance(text, str) and text.startswith('ordinal:')


def date_of(text):
    if is_ordinal(text):
        return _Ordinal(int(text.split(':', 1)[1]))
    return parse(text, fuzzy=True).date()


def _month_span(year, month):
    last = calendar.monthrange(year, month)[1]
    return (year, month, 1), (year, month, last)


def _year_span(year):
    return (year, 1, 1), (year, 12, 31)


def _as_date(parts):
    'Accept either a (year, month, day) tuple or an already built date.'
    if parts is None:
        return None
    if isinstance(parts, tuple):
        return '%04d-%02d-%02d' % parts
    return parts.isoformat()


def _shift(day, count, unit):
    if unit == 'days':
        return day + timedelta(days=count)
    if unit == 'weeks':
        return day + timedelta(weeks=count)
    if unit == 'months':
        return day + relativedelta(months=count)
    return day + relativedelta(years=count)


def _count_of(phrase):
    'Parse the count part of "three days", "a couple of weeks", "5 months".'
    stripped = ' '.join(word for word in phrase.split() if word != 'of')
    if not stripped:
        return 1
    if stripped.isdigit():
        return int(stripped)
    if stripped in _COUNT:
        return _COUNT[stripped]
    if stripped.startswith('a couple') or stripped.startswith('couple'):
        return 2
    return None


def anchor(expression, recorded_at, role='event'):
    'Normalize one time expression against the reporting time.'
    day = date_of(recorded_at)
    raw = (expression or '').strip()
    phrase = ' '.join(raw.lower().strip(' .,').split())
    lo = hi = None
    precision = 'unknown'
    inferred = None

    def single(offset, kind='day'):
        nonlocal lo, hi, precision
        lo = hi = day + timedelta(days=offset)
        precision = kind

    words = phrase.split()

    if phrase in _VAGUE:
        pass
    elif len(phrase) == 10 and phrase[4] == '-' and phrase[7] == '-' and phrase.replace('-', '').isdigit():
        lo = hi = date_of(phrase)
        precision = 'day'
    elif phrase in _EXACT:
        single(*_EXACT[phrase])
    elif phrase in _RANGES:
        kind = _RANGES[phrase]
        if kind == 'last_weekend':
            end = day - timedelta(days=day.weekday() + 1)
            lo, hi = end - timedelta(days=1), end
        elif kind == 'last_week':
            end = day - timedelta(days=day.weekday() + 1)
            lo, hi = end - timedelta(days=6), end
        elif kind == 'this_week':
            lo = day - timedelta(days=day.weekday())
            hi = day + timedelta(days=6 - day.weekday())
        elif kind == 'next_week':
            start = day + timedelta(days=7 - day.weekday())
            lo, hi = start, start + timedelta(days=6)
        elif kind == 'last_month':
            start = day.replace(day=1) - relativedelta(months=1)
            lo, hi = _month_span(start.year, start.month)
        elif kind == 'next_month':
            start = day.replace(day=1) + relativedelta(months=1)
            lo, hi = _month_span(start.year, start.month)
        elif kind == 'this_month':
            lo, hi = _month_span(day.year, day.month)
        elif kind == 'last_year':
            lo, hi = _year_span(day.year - 1)
        elif kind == 'next_year':
            lo, hi = _year_span(day.year + 1)
        else:
            lo, hi = _year_span(day.year)
        if kind in ('last_weekend', 'last_week', 'this_week', 'next_week'):
            precision = 'range'
        else:
            precision = kind.split('_')[-1]
    elif words and words[0] in ('last', 'next', 'this', 'past') and words[-1] in _WEEKDAY:


        lead = words[0]
        if lead == 'this' and len(words) > 1 and words[1] in ('past', 'last'):
            lead = 'past'
        target = _WEEKDAY[words[-1]]
        if lead in ('last', 'past'):
            offset = -((day.weekday() - target) % 7 or 7)
        elif lead == 'next':
            offset = (target - day.weekday()) % 7 or 7
        else:
            offset = target - day.weekday()
        single(offset)
    elif phrase.endswith(' ago') and len(words) >= 3:
        head = words[:-1]
        unit = _UNIT.get(head[-1])
        if unit:
            quantity = ' '.join(head[:-1])
            count = _count_of(quantity)
            if count is not None:
                lo = hi = _shift(day, -count, unit)
                precision = 'day' if unit == 'days' else 'range'
            elif _is_approximate(quantity):
                inferred = 'approximate_quantity'
    elif words and words[0] == 'in' and len(words) >= 3:
        unit = _UNIT.get(words[-1])
        if unit:
            quantity = ' '.join(words[1:-1])
            count = _count_of(quantity)
            if count is not None:
                lo = hi = _shift(day, count, unit)
                precision = 'day' if unit == 'days' else 'range'
            elif _is_approximate(quantity):
                inferred = 'approximate_quantity'
    else:
        bare = [word for word in words if word not in ('in', 'during', 'on', 'the')]
        if len(bare) == 1 and bare[0] in _MONTH:


            precision = 'month'
            inferred = 'year_unresolved'
        elif len(bare) == 1 and bare[0].isdigit() and len(bare[0]) == 4:
            lo, hi = _year_span(int(bare[0]))
            precision = 'year'
        elif len(bare) == 2 and bare[0] in _MONTH and bare[1].isdigit() and len(bare[1]) == 4:
            lo, hi = _month_span(int(bare[1]), _MONTH[bare[0]])
            precision = 'month'

    text = ('%s (as of %s)' % (raw, day.isoformat())) if raw else ('Unspecified event time; reported %s' % day.isoformat())
    if inferred == 'year_unresolved':
        text += ' [month stated without a year; no year is assumed]'
    elif inferred == 'approximate_quantity':
        text += ' [approximate quantity; no exact date is asserted]'
    return {'raw_expression': raw, 'recorded_at': recorded_at, 'role': role,
            'semantic_anchor': text, 'start': _as_date(lo), 'end': _as_date(hi),
            'precision': precision, 'inferred': inferred}
