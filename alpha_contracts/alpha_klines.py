from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional, Sequence, Tuple, Union

Number = Union[int, float]


@dataclass
class Candle:
	open_time_ms: int
	open: float
	high: float
	low: float
	close: float
	volume: Optional[float] = None
	quote: Optional[str] = None
	pair_id: Optional[str] = None

	@property
	def open_time_iso(self) -> str:
		return datetime.fromtimestamp(self.open_time_ms / 1000.0, tz=timezone.utc).isoformat()

	def as_list(self) -> List[Union[int, float, str, None]]:
		return [
			self.open_time_ms,
			self.open,
			self.high,
			self.low,
			self.close,
			self.volume if self.volume is not None else 0.0,
			self.quote,
			self.pair_id,
		]


def _is_timestamp_like(value: Any) -> Optional[int]:
	"""
	Return epoch ms if value looks like a timestamp in seconds or ms.
	"""
	try:
		if isinstance(value, (int, float)):
			val = int(value)
			# milliseconds
			if 10**10 < val < 10**14:
				return val
			# seconds
			if 10**8 < val < 10**10:
				return val * 1000
	except Exception:
		return None
	return None


def _parse_row_to_candle(row: Sequence[Any]) -> Optional[Candle]:
	"""
	Attempt to parse a single kline-like row to Candle.
	Accept common shapes: [openTime, open, high, low, close, volume? ...]
	"""
	if not isinstance(row, (list, tuple)) or len(row) < 5:
		return None
	ts_ms = _is_timestamp_like(row[0])
	if ts_ms is None:
		return None
	try:
		open_px = float(row[1])
		high_px = float(row[2])
		low_px = float(row[3])
		close_px = float(row[4])
		vol = float(row[5]) if len(row) > 5 and isinstance(row[5], (int, float, str)) else None
		return Candle(
			open_time_ms=ts_ms,
			open=open_px,
			high=high_px,
			low=low_px,
			close=close_px,
			volume=vol,
		)
	except Exception:
		return None


def extract_earliest_candle_from_json(obj: Any) -> Optional[Candle]:
	"""
	Walk a nested JSON object and find the earliest candle row by scanning
	for array-of-arrays where each inner array is kline-like. From all series found,
	select the minimum openTime.
	"""
	earliest: Optional[Candle] = None

	def consider_series(series: Sequence[Any]) -> None:
		nonlocal earliest
		if not series or not isinstance(series, (list, tuple)):
			return
		# If it's a series of rows
		if isinstance(series[0], (list, tuple)):
			for row in series:
				c = _parse_row_to_candle(row)
				if c is None:
					continue
				if earliest is None or c.open_time_ms < earliest.open_time_ms:
					earliest = c
		else:
			# It could be a single row
			c = _parse_row_to_candle(series)
			if c:
				if earliest is None or c.open_time_ms < earliest.open_time_ms:
					earliest = c

	def walk(x: Any) -> None:
		if isinstance(x, dict):
			# Try to pick up quote or pair id from adjacent keys if present
			# but without strong assumptions; we attach only when we already have an earliest.
			for v in x.values():
				walk(v)
		elif isinstance(x, (list, tuple)):
			# Heuristic: if first element is a kline-like row or the list contains rows, consider it.
			if x and isinstance(x[0], (list, tuple)):
				# Check if the first row is kline-like
				if _parse_row_to_candle(x[0]) is not None:
					consider_series(x)
					return
			else:
				# Maybe a single row list
				if _parse_row_to_candle(x) is not None:
					consider_series(x)
					return
			# Otherwise keep walking
			for v in x:
				walk(v)

	walk(obj)
	return earliest





