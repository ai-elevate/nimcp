#!/usr/bin/env python3
"""
NIMCP Streaming Training Pipeline
==================================

WHAT: Memory-efficient training on massive datasets using streaming
WHY:  Avoid downloading 6TB+ datasets - train incrementally on batches
HOW:  Stream dataset → Process batch → Train → Discard → Repeat

Features:
- Streaming mode (no full download required)
- Configurable batch sizes (20-30 parquet files at a time)
- Automatic checkpoint saving
- Resume from checkpoint support
- Progress tracking
- Memory-efficient cleanup
"""

import hashlib
import json
import os
import sys
import time
import gc
import shutil
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Iterator
from dataclasses import dataclass
from datasets import load_dataset, IterableDataset
import nimcp

@dataclass
class StreamConfig:
    """Configuration for streaming training"""
    batch_size: int = 1000  # Examples per training batch
    parquet_batch_size: int = 25  # Number of parquet files to buffer
    checkpoint_interval: int = 10000  # Save checkpoint every N examples
    max_examples_per_dataset: Optional[int] = None  # Limit examples (for testing)
    temp_dir: Path = Path("/tmp/nimcp_streaming")
    checkpoint_dir: Path = Path("checkpoints")

@dataclass
class TrainingProgress:
    """Track training progress for checkpointing"""
    dataset_name: str
    examples_processed: int = 0
    batches_completed: int = 0
    last_checkpoint: int = 0
    start_time: float = 0.0

    def save(self, checkpoint_dir: Path):
        """Save progress to checkpoint file"""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / f"{self.dataset_name}_progress.json"
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'dataset_name': self.dataset_name,
                'examples_processed': self.examples_processed,
                'batches_completed': self.batches_completed,
                'last_checkpoint': self.last_checkpoint,
                'timestamp': time.time()
            }, f, indent=2)

    @classmethod
    def load(cls, dataset_name: str, checkpoint_dir: Path):
        """Load progress from checkpoint file"""
        checkpoint_file = checkpoint_dir / f"{dataset_name}_progress.json"
        if not checkpoint_file.exists():
            return cls(dataset_name=dataset_name, start_time=time.time())

        with open(checkpoint_file) as f:
            data = json.load(f)

        progress = cls(
            dataset_name=data['dataset_name'],
            examples_processed=data['examples_processed'],
            batches_completed=data['batches_completed'],
            last_checkpoint=data['last_checkpoint'],
            start_time=time.time()
        )
        print(f"  ↻ Resuming from checkpoint: {progress.examples_processed} examples processed")
        return progress


class StreamingDatasetProcessor:
    """
    WHAT: Process datasets in streaming mode with batch training
    WHY:  Handle massive datasets without downloading everything
    HOW:  Stream → Buffer → Train → Cleanup
    """

    def __init__(self, brain, config: StreamConfig, num_inputs: int = 128, hf_token: str = None):
        self.brain = brain
        self.config = config
        self.num_inputs = num_inputs
        self.hf_token = hf_token
        self.config.temp_dir.mkdir(parents=True, exist_ok=True)
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def load_streaming_dataset(self, dataset_config: Dict):
        """
        WHAT: Load a HuggingFace dataset in streaming mode
        WHY:  Avoid downloading entire dataset; returns raw iterator (NOT a generator)
        HOW:  Use streaming=True, return the IterableDataset directly
        """
        hf_dataset = dataset_config['hf_dataset']
        hf_subset = dataset_config.get('hf_subset', None)

        print(f"  🔄 Streaming from HuggingFace: {hf_dataset}" +
              (f" ({hf_subset})" if hf_subset else ""))

        kwargs = dict(streaming=True)
        if self.hf_token:
            kwargs['token'] = self.hf_token

        # Use explicit split if specified (e.g. "en" for language-split datasets)
        explicit_split = dataset_config.get('hf_split')
        splits_to_try = [explicit_split] if explicit_split else ['train', 'test', 'validation', 'dev']

        for split in splits_to_try:
            try:
                if hf_subset:
                    dataset = load_dataset(hf_dataset, hf_subset, split=split, **kwargs)
                else:
                    dataset = load_dataset(hf_dataset, split=split, **kwargs)
                print(f"  📂 Using split: {split}")
                return dataset
            except (ValueError, Exception) as split_err:
                err_str = str(split_err)
                if 'Bad split' in err_str or 'does not exist' in err_str:
                    continue
                print(f"  ✗ Streaming failed: {split_err}")
                return None

        print(f"  ✗ No usable split found")
        return None

    def load_local_dataset(self, dataset_config: Dict):
        """Load a local JSONL dataset as a streaming iterator.

        Expects dataset_config to have 'local_path' pointing to a directory
        containing train.jsonl (each line: {"text": "...", "label": "..."}).
        """
        local_path = dataset_config.get("local_path", "")
        jsonl_file = Path(local_path) / "train.jsonl"

        if not jsonl_file.exists():
            print(f"  ✗ Local dataset not found: {jsonl_file}")
            return None

        print(f"  📁 Loading local dataset: {jsonl_file}")
        try:
            dataset = load_dataset("json", data_files=str(jsonl_file),
                                   split="train", streaming=True)
            return dataset
        except Exception as e:
            print(f"  ✗ Failed to load local dataset: {e}")
            return None

    def load_api_stream(self, dataset_config: Dict):
        """Load a streaming dataset from a REST API source.

        Dispatches to source-specific generators based on dataset_config['type'].
        Each generator yields dicts with at least 'text' and 'label' keys,
        compatible with extract_features_and_label().
        All data is streamed over HTTP — nothing saved to disk.
        """
        source_type = dataset_config.get("type", "")
        loader = {
            "wikipedia": self._stream_wikipedia,
            "arxiv": self._stream_arxiv,
            "stackexchange": self._stream_stackexchange,
            "pubmed": self._stream_pubmed,
            "gutenberg": self._stream_gutenberg,
            "conceptnet": self._stream_conceptnet,
            "news_rss": self._stream_news_rss,
        }.get(source_type)

        if loader is None:
            print(f"  ✗ Unknown API source type: {source_type}")
            return None

        print(f"  🌐 Streaming from API: {source_type} "
              f"({dataset_config.get('name', '?')})")
        return loader(dataset_config)

    def _api_get_json(self, url: str, timeout: int = 30) -> Optional[dict]:
        """Fetch JSON from a URL. Returns None on failure."""
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "NIMCP-Athena/1.0 (training pipeline)"
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"    [API error: {e}]", end="", flush=True)
            return None

    def _api_get_text(self, url: str, timeout: int = 30) -> Optional[str]:
        """Fetch raw text from a URL. Returns None on failure."""
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "NIMCP-Athena/1.0 (training pipeline)"
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception:
            return None

    # --- Wikipedia API (random articles, endless stream) ---

    def _stream_wikipedia(self, config: Dict):
        """Stream random Wikipedia articles via the MediaWiki API.

        Config options:
          language: wiki language code (default "en")
          max_examples: stop after N articles (default 50000)
        """
        lang = config.get("language", "en")
        max_ex = config.get("max_examples", 50_000)
        base = f"https://{lang}.wikipedia.org/w/api.php"
        yielded = 0

        while yielded < max_ex:
            # Fetch 20 random articles per request
            params = urllib.parse.urlencode({
                "action": "query",
                "format": "json",
                "generator": "random",
                "grnnamespace": "0",
                "grnlimit": "20",
                "prop": "extracts",
                "explaintext": "true",
                "exlimit": "20",
            })
            data = self._api_get_json(f"{base}?{params}")
            if not data:
                time.sleep(2)
                continue

            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                text = page.get("extract", "")
                title = page.get("title", "")
                if not text or len(text) < 100:
                    continue
                yield {
                    "text": f"{title}\n\n{text[:3000]}",
                    "label": hash(title) % 8,
                }
                yielded += 1
                if yielded >= max_ex:
                    return

            # Respect rate limits
            time.sleep(0.5)

    # --- arXiv API (stream papers by category) ---

    def _stream_arxiv(self, config: Dict):
        """Stream arXiv paper abstracts via the arXiv API.

        Config options:
          category: arXiv category (default "cs.AI")
          max_examples: stop after N papers (default 50000)
        """
        category = config.get("category", "cs.AI")
        max_ex = config.get("max_examples", 50_000)
        base = "http://export.arxiv.org/api/query"
        batch_size = 100
        start = 0
        yielded = 0
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        while yielded < max_ex:
            params = urllib.parse.urlencode({
                "search_query": f"cat:{category}",
                "start": start,
                "max_results": batch_size,
                "sortBy": "lastUpdatedDate",
                "sortOrder": "descending",
            })
            xml_text = self._api_get_text(f"{base}?{params}")
            if not xml_text:
                time.sleep(5)
                start += batch_size
                continue

            try:
                root = ET.fromstring(xml_text)
            except ET.ParseError:
                time.sleep(5)
                start += batch_size
                continue

            entries = root.findall("atom:entry", ns)
            if not entries:
                break  # No more results

            for entry in entries:
                title_el = entry.find("atom:title", ns)
                summary_el = entry.find("atom:summary", ns)
                title = title_el.text.strip() if title_el is not None and title_el.text else ""
                abstract = summary_el.text.strip() if summary_el is not None and summary_el.text else ""
                if not abstract or len(abstract) < 50:
                    continue

                # Extract primary category
                prim_cat = entry.find("arxiv:primary_category",
                                     {"arxiv": "http://arxiv.org/schemas/atom"})
                cat_label = prim_cat.get("term", category) if prim_cat is not None else category

                yield {
                    "text": f"{title}\n\n{abstract}",
                    "label": hash(cat_label) % 8,
                }
                yielded += 1
                if yielded >= max_ex:
                    return

            start += batch_size
            # arXiv rate limit: 1 request per 3 seconds
            time.sleep(3)

    # --- StackExchange API (stream Q&A pairs) ---

    def _stream_stackexchange(self, config: Dict):
        """Stream StackExchange questions+answers via the API.

        Config options:
          site: SE site name (default "stackoverflow")
          tagged: comma-separated tags to filter (optional)
          max_examples: stop after N Q&A pairs (default 50000)
        """
        site = config.get("site", "stackoverflow")
        tagged = config.get("tagged", "")
        max_ex = config.get("max_examples", 50_000)
        base = "https://api.stackexchange.com/2.3"
        page = 1
        yielded = 0

        while yielded < max_ex:
            params = {
                "order": "desc",
                "sort": "votes",
                "site": site,
                "filter": "withbody",
                "pagesize": "100",
                "page": str(page),
            }
            if tagged:
                params["tagged"] = tagged
            url = f"{base}/questions?{urllib.parse.urlencode(params)}"
            data = self._api_get_json(url)
            if not data:
                time.sleep(5)
                page += 1
                continue

            items = data.get("items", [])
            if not items:
                break

            for item in items:
                title = item.get("title", "")
                body = item.get("body", "")
                # Strip HTML tags (basic)
                import re
                body_text = re.sub(r"<[^>]+>", " ", body).strip()
                tags = item.get("tags", [])
                score = item.get("score", 0)

                if not body_text or len(body_text) < 50:
                    continue

                tag_str = ", ".join(tags[:5]) if tags else ""
                text = f"[{tag_str}] {title}\n\n{body_text[:2000]}"
                yield {
                    "text": text,
                    "label": min(score, 99),
                }
                yielded += 1
                if yielded >= max_ex:
                    return

            if not data.get("has_more", False):
                break

            page += 1
            # SE API: max 30 requests/sec without key, be conservative
            time.sleep(2)

    # --- PubMed API (stream biomedical abstracts) ---

    def _stream_pubmed(self, config: Dict):
        """Stream PubMed abstracts via NCBI E-utilities.

        Config options:
          query: PubMed search query (default "machine learning")
          max_examples: stop after N abstracts (default 50000)
        """
        query = config.get("query", "machine learning")
        max_ex = config.get("max_examples", 50_000)
        base_search = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        base_fetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        batch_size = 200
        retstart = 0
        yielded = 0

        while yielded < max_ex:
            # Step 1: Search for PMIDs
            search_params = urllib.parse.urlencode({
                "db": "pubmed",
                "term": query,
                "retmax": batch_size,
                "retstart": retstart,
                "retmode": "json",
                "sort": "relevance",
            })
            search_data = self._api_get_json(f"{base_search}?{search_params}")
            if not search_data:
                time.sleep(3)
                retstart += batch_size
                continue

            id_list = search_data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                break

            # Step 2: Fetch abstracts for these PMIDs
            fetch_params = urllib.parse.urlencode({
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "xml",
                "rettype": "abstract",
            })
            xml_text = self._api_get_text(f"{base_fetch}?{fetch_params}")
            if not xml_text:
                time.sleep(3)
                retstart += batch_size
                continue

            try:
                root = ET.fromstring(xml_text)
            except ET.ParseError:
                retstart += batch_size
                continue

            for article in root.findall(".//PubmedArticle"):
                title_el = article.find(".//ArticleTitle")
                abstract_el = article.find(".//AbstractText")
                title = title_el.text if title_el is not None and title_el.text else ""
                abstract = abstract_el.text if abstract_el is not None and abstract_el.text else ""

                if not abstract or len(abstract) < 50:
                    continue

                # Extract MeSH terms for label
                mesh_terms = [m.text for m in article.findall(".//MeshHeading/DescriptorName")
                              if m.text]
                label = hash(mesh_terms[0]) % 8 if mesh_terms else 0

                yield {
                    "text": f"{title}\n\n{abstract}",
                    "label": label,
                }
                yielded += 1
                if yielded >= max_ex:
                    return

            retstart += batch_size
            # NCBI: max 3 requests/sec without API key
            time.sleep(1)

    # --- Project Gutenberg (stream full books by ID range) ---

    def _stream_gutenberg(self, config: Dict):
        """Stream books from Project Gutenberg via direct text URLs.

        Config options:
          start_id: first book ID to try (default 1)
          end_id: last book ID to try (default 70000)
          max_examples: max chunks to yield (default 50000)
          chunk_size: characters per chunk (default 2000)
        """
        start_id = config.get("start_id", 1)
        end_id = config.get("end_id", 70000)
        max_ex = config.get("max_examples", 50_000)
        chunk_size = config.get("chunk_size", 2000)
        yielded = 0

        import random
        # Randomize order to get variety
        ids = list(range(start_id, end_id + 1))
        random.shuffle(ids)

        for book_id in ids:
            if yielded >= max_ex:
                return

            # Gutenberg plain text URL pattern
            url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
            text = self._api_get_text(url)
            if not text or len(text) < 500:
                continue

            # Strip Gutenberg header/footer
            start_marker = "*** START OF"
            end_marker = "*** END OF"
            start_idx = text.find(start_marker)
            end_idx = text.find(end_marker)
            if start_idx >= 0:
                text = text[text.index("\n", start_idx) + 1:]
            if end_idx >= 0:
                text = text[:end_idx]

            # Chunk into training examples
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size].strip()
                if len(chunk) < 100:
                    continue
                yield {
                    "text": chunk,
                    "label": book_id % 8,
                }
                yielded += 1
                if yielded >= max_ex:
                    return

            # Be polite to Gutenberg servers
            time.sleep(1)

    # --- ConceptNet API (stream semantic relationships) ---

    def _stream_conceptnet(self, config: Dict):
        """Stream semantic relationships from ConceptNet API.

        Config options:
          language: language code (default "en")
          max_examples: stop after N relations (default 50000)
        """
        lang = config.get("language", "en")
        max_ex = config.get("max_examples", 50_000)
        base = "https://api.conceptnet.io"
        yielded = 0

        # Seed concepts to start crawling from
        seeds = [
            "learning", "knowledge", "science", "nature", "computer",
            "brain", "language", "math", "history", "art", "music",
            "food", "animal", "water", "energy", "time", "space",
            "emotion", "health", "society", "technology", "earth",
        ]
        visited = set()
        to_visit = list(seeds)

        while to_visit and yielded < max_ex:
            concept = to_visit.pop(0)
            if concept in visited:
                continue
            visited.add(concept)

            url = f"{base}/c/{lang}/{concept}?limit=100"
            data = self._api_get_json(url)
            if not data:
                time.sleep(1)
                continue

            edges = data.get("edges", [])
            for edge in edges:
                rel = edge.get("rel", {}).get("label", "")
                start_label = edge.get("start", {}).get("label", "")
                end_label = edge.get("end", {}).get("label", "")
                weight = edge.get("weight", 1.0)

                if not rel or not start_label or not end_label:
                    continue

                text = f"{start_label} {rel} {end_label}"
                yield {
                    "text": text,
                    "label": hash(rel) % 8,
                }
                yielded += 1
                if yielded >= max_ex:
                    return

                # Add discovered concepts to crawl queue
                for lbl in (start_label, end_label):
                    clean = lbl.lower().replace(" ", "_")
                    if clean not in visited and len(to_visit) < 10000:
                        to_visit.append(clean)

            time.sleep(0.5)

    # --- News RSS (continuous live news stream) ---

    # Default feeds — broad coverage, no API keys needed
    _DEFAULT_NEWS_FEEDS = [
        # General / World
        ("https://feeds.reuters.com/reuters/topNews", "world"),
        ("https://feeds.reuters.com/reuters/worldNews", "world"),
        ("https://feeds.bbci.co.uk/news/rss.xml", "world"),
        ("https://feeds.bbci.co.uk/news/world/rss.xml", "world"),
        ("https://rss.nytimes.com/services/xml/rss/nyt/World.xml", "world"),
        ("https://www.aljazeera.com/xml/rss/all.xml", "world"),
        ("https://feeds.npr.org/1001/rss.xml", "world"),
        ("https://www.theguardian.com/world/rss", "world"),
        # Science
        ("https://feeds.bbci.co.uk/news/science_and_environment/rss.xml", "science"),
        ("https://rss.nytimes.com/services/xml/rss/nyt/Science.xml", "science"),
        ("https://www.nature.com/nature.rss", "science"),
        ("https://feeds.arstechnica.com/arstechnica/science", "science"),
        # Technology
        ("https://feeds.bbci.co.uk/news/technology/rss.xml", "technology"),
        ("https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml", "technology"),
        ("https://feeds.arstechnica.com/arstechnica/index", "technology"),
        ("https://hnrss.org/frontpage", "technology"),
        # Business / Finance
        ("https://feeds.reuters.com/reuters/businessNews", "finance"),
        ("https://rss.nytimes.com/services/xml/rss/nyt/Business.xml", "finance"),
        ("https://feeds.bbci.co.uk/news/business/rss.xml", "finance"),
        # Health / Medicine
        ("https://feeds.bbci.co.uk/news/health/rss.xml", "medicine"),
        ("https://rss.nytimes.com/services/xml/rss/nyt/Health.xml", "medicine"),
    ]

    def _stream_news_rss(self, config: Dict):
        """Stream live news articles from RSS feeds — continuous, never finishes.

        Polls a rotating set of RSS feeds, deduplicates by URL, and yields
        new articles as training examples. Designed to run indefinitely so
        the instructor thread stays alive and keeps learning current events.

        Config options:
          feeds: list of {"url": "...", "category": "..."} (optional, uses defaults)
          poll_interval_s: seconds between full poll cycles (default 300 = 5 min)
          max_examples: stop after N articles (default 0 = unlimited/continuous)
        """
        import re as _re

        custom_feeds = config.get("feeds")
        if custom_feeds:
            feeds = [(f["url"], f.get("category", "news")) for f in custom_feeds]
        else:
            feeds = list(self._DEFAULT_NEWS_FEEDS)

        poll_interval = config.get("poll_interval_s", 300)
        max_ex = config.get("max_examples", 0)  # 0 = unlimited
        seen_urls = set()
        yielded = 0

        while True:
            cycle_count = 0

            for feed_url, category in feeds:
                xml_text = self._api_get_text(feed_url, timeout=15)
                if not xml_text:
                    continue

                try:
                    root = ET.fromstring(xml_text)
                except ET.ParseError:
                    continue

                # RSS 2.0: channel/item
                items = root.findall(".//item")
                # Atom: entry
                if not items:
                    ns = {"atom": "http://www.w3.org/2005/Atom"}
                    items = root.findall("atom:entry", ns)

                for item in items:
                    # Extract link for dedup
                    link_el = item.find("link")
                    if link_el is not None:
                        link = link_el.text or link_el.get("href", "")
                    else:
                        link = ""

                    if link in seen_urls:
                        continue
                    if link:
                        seen_urls.add(link)

                    # Extract title
                    title_el = item.find("title")
                    title = title_el.text.strip() if title_el is not None and title_el.text else ""

                    # Extract description/summary
                    desc = ""
                    for tag in ("description", "summary", "content"):
                        el = item.find(tag)
                        if el is not None and el.text:
                            desc = el.text.strip()
                            break
                    # Atom content:encoded
                    if not desc:
                        ce = item.find("{http://purl.org/rss/1.0/modules/content/}encoded")
                        if ce is not None and ce.text:
                            desc = ce.text.strip()

                    if not title and not desc:
                        continue

                    # Strip HTML tags
                    desc_clean = _re.sub(r"<[^>]+>", " ", desc).strip()

                    # Extract publication date if available
                    pub_date = ""
                    for tag in ("pubDate", "published", "updated"):
                        pd_el = item.find(tag)
                        if pd_el is not None and pd_el.text:
                            pub_date = pd_el.text.strip()
                            break

                    text = f"[{category}] {title}"
                    if pub_date:
                        text += f" ({pub_date})"
                    text += f"\n\n{desc_clean[:2000]}"

                    yield {
                        "text": text,
                        "label": hash(category) % 8,
                    }
                    yielded += 1
                    cycle_count += 1

                    if max_ex > 0 and yielded >= max_ex:
                        return

                # Brief pause between feeds to avoid hammering
                time.sleep(1)

            # Cap seen_urls to prevent unbounded memory growth
            if len(seen_urls) > 100_000:
                # Keep most recent half
                seen_urls = set(list(seen_urls)[-50_000:])

            if cycle_count > 0:
                print(f"    [News] Cycle complete: {cycle_count} new articles, "
                      f"{yielded} total", flush=True)

            # Wait before next poll cycle
            time.sleep(poll_interval)

    def encode_text(self, text: str, num_features: int) -> List[float]:
        """Encode text into a fixed-size feature vector.

        Uses benchmark_datasets.text_to_features for consistent encoding
        across the entire training pipeline.
        """
        try:
            from benchmark_datasets import text_to_features
            return text_to_features(text, num_features)
        except ImportError:
            pass
        # Inline fallback if benchmark_datasets unavailable
        features = [0.0] * num_features
        text_lower = text.lower().strip()
        if not text_lower:
            return features
        q = num_features // 4
        for ch in text_lower:
            features[ord(ch) % q] += 1.0
        for i in range(len(text_lower) - 1):
            bigram = text_lower[i:i + 2]
            h = int(hashlib.md5(bigram.encode()).hexdigest(), 16)
            features[q + (h % q)] += 1.0
        words = text_lower.split()
        for wi, word in enumerate(words):
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            features[2 * q + (h % q)] += 1.0
            features[2 * q + ((h >> 16) % q)] += (wi + 1) * 0.05
        for i in range(len(words) - 1):
            pair = f"{words[i]} {words[i+1]}"
            h = int(hashlib.md5(pair.encode()).hexdigest(), 16)
            features[2 * q + (h % q)] += 0.7
        mx = max(features) if features else 1.0
        if mx > 0:
            features = [v / mx for v in features]
        return features

    def _extract_text_and_label(self, example: Dict, domain: str) -> Optional[tuple]:
        """Extract raw text and label from a dataset example based on its
        schema.  Returns (text, label_int) or None if the example is empty."""

        # --- MMLU format (cais/mmlu): question + choices + answer index ---
        if 'question' in example and 'choices' in example and 'answer' in example:
            q = example['question']
            choices = example['choices']
            if isinstance(choices, list):
                choices_str = ' '.join(f"({chr(65+i)}) {c}" for i, c in enumerate(choices))
            else:
                choices_str = str(choices)
            text = f"{q} {choices_str}"
            label = int(example['answer']) if isinstance(example['answer'], (int, float)) else 0
            return (text, label)

        # --- SQuAD format: context + question + answers ---
        if 'context' in example and 'question' in example:
            text = f"{example['question']} {example['context']}"
            answers = example.get('answers', {})
            if isinstance(answers, dict):
                ans_text = ' '.join(answers.get('text', []))
            elif isinstance(answers, list):
                ans_text = ' '.join(str(a) for a in answers)
            else:
                ans_text = str(answers)
            text = f"{text} {ans_text}"
            label = hash(ans_text) % 8 if ans_text.strip() else 0
            return (text, label)

        # --- CommonsenseQA: question + choices ---
        if 'question' in example and 'choices' in example and 'answerKey' in example:
            q = example['question']
            ch = example['choices']
            if isinstance(ch, dict):
                labels_list = ch.get('label', [])
                texts_list = ch.get('text', [])
                choices_str = ' '.join(f"({l}) {t}" for l, t in zip(labels_list, texts_list))
            else:
                choices_str = str(ch)
            text = f"{q} {choices_str}"
            ak = example['answerKey']
            label = ord(ak) - ord('A') if isinstance(ak, str) and ak.isalpha() else 0
            return (text, label)

        # --- Ethics format (hendrycks/ethics): input + label ---
        if 'input' in example and 'label' in example:
            text = str(example['input'])
            label = int(example['label']) if isinstance(example['label'], (int, float)) else 0
            return (text, label)

        # --- Moral stories: multiple text fields ---
        if 'norm' in example or 'situation' in example or 'moral_action' in example:
            parts = []
            for key in ('norm', 'situation', 'intention', 'moral_action', 'immoral_action'):
                val = example.get(key, '')
                if val:
                    parts.append(str(val))
            text = ' '.join(parts)
            label = int(example.get('label', 0)) if 'label' in example else 0
            return (text, label)

        # --- MMMU format: question + options ---
        if 'question' in example and 'options' in example:
            q = str(example['question'])
            opts = example['options']
            if isinstance(opts, list):
                opts_str = ' '.join(f"({chr(65+i)}) {o}" for i, o in enumerate(opts))
            else:
                opts_str = str(opts)
            text = f"{q} {opts_str}"
            ans = example.get('answer', '')
            label = ord(str(ans)) - ord('A') if isinstance(ans, str) and ans.isalpha() else 0
            return (text, label)

        # --- Programming: content/code ---
        if 'content' in example or 'code' in example:
            text = example.get('content', example.get('code', ''))
            lang = example.get('language', '')
            if lang:
                text = f"{lang}: {text}"
            label = hash(lang) % 8 if lang else 0
            return (text, label)

        # --- LiveCodeBench: question_content ---
        if 'question_content' in example:
            text = str(example['question_content'])
            label = hash(example.get('question_id', '')) % 8
            return (text, label)

        # --- Plain text (wikitext, etc.) ---
        if 'text' in example:
            text = str(example['text'])
            if not text.strip():
                return None
            label = 0
            return (text, label)

        # --- Social chem: action + rot (rule of thumb) ---
        if 'action' in example or 'rot' in example:
            parts = [str(example.get('action', '')), str(example.get('rot', ''))]
            text = ' '.join(p for p in parts if p)
            label = int(example.get('rot-judgment', 0)) if 'rot-judgment' in example else 0
            return (text, label)

        # --- HellaSwag format: ctx + endings + label ---
        if 'ctx' in example and 'endings' in example:
            ctx = str(example.get('ctx', ''))
            endings = example.get('endings', [])
            if isinstance(endings, list):
                endings_str = ' '.join(f"({i}) {e}" for i, e in enumerate(endings))
            else:
                endings_str = str(endings)
            text = f"{ctx} {endings_str}"
            label = int(example.get('label', 0)) if example.get('label') is not None else 0
            return (text, label)

        # --- OpenOrca format: system_prompt + question + response ---
        if 'system_prompt' in example and 'question' in example and 'response' in example:
            text = f"{example['system_prompt']} {example['question']} {example['response']}"
            label = hash(example.get('id', '')) % 8
            return (text, label)

        # --- Prosocial dialog (allenai/prosocial-dialog) ---
        if 'rots' in example and 'safety_label' in example:
            parts = [str(example.get('context', '')), str(example.get('response', ''))]
            rots = example.get('rots', [])
            if isinstance(rots, list):
                parts.extend(str(r) for r in rots)
            text = ' '.join(p for p in parts if p)
            sl = example.get('safety_label', '')
            label = 0 if sl == '__casual__' else 1 if sl == '__needs_caution__' else 2
            return (text, label)

        # --- HH-RLHF format (Anthropic/hh-rlhf): chosen + rejected ---
        if 'chosen' in example and 'rejected' in example:
            text = str(example['chosen'])
            label = 0  # chosen = good
            return (text, label)

        # --- CultureBank format (SALT-NLP/CultureBank) ---
        if 'cultural_group' in example or 'actor_behavior' in example:
            parts = []
            for key in ('cultural_group', 'context', 'goal', 'actor_behavior', 'topic'):
                val = example.get(key, '')
                if val:
                    parts.append(str(val))
            text = ' '.join(parts)
            agreement = example.get('agreement', '')
            label = hash(str(agreement)) % 8 if agreement else 0
            return (text, label)

        # --- Code contests (deepmind/code_contests) ---
        if 'description' in example and 'solutions' in example:
            text = str(example.get('description', ''))
            sols = example.get('solutions', {})
            if isinstance(sols, dict):
                sol_list = sols.get('solution', [])
                if isinstance(sol_list, list) and sol_list:
                    text = f"{text}\n{sol_list[0][:1000]}"
            label = int(example.get('difficulty', 0)) if 'difficulty' in example else 0
            return (text, label)

        # --- PalmX / query-answer format ---
        if 'query' in example:
            text = str(example['query'])
            ans = example.get('answer', '')
            if ans:
                text = f"{text} {ans}"
            label = 0
            return (text, label)

        # --- TinyStories format (roneneldan/TinyStories): story field ---
        if 'story' in example:
            text = str(example['story'])
            if not text.strip():
                return None
            label = 0
            return (text, label)

        # --- BookSum format (kmfoda/booksum): summary_text + chapter ---
        if 'summary_text' in example or 'summary' in example:
            text = str(example.get('summary_text', example.get('summary', '')))
            chapter = example.get('chapter', example.get('book_name', ''))
            if chapter:
                text = f"{chapter}: {text}"
            if not text.strip():
                return None
            label = 0
            return (text, label)

        # --- Fallback: concatenate all string fields ---
        parts = []
        label = 0
        for key, value in example.items():
            if isinstance(value, str) and value.strip():
                parts.append(value)
            elif isinstance(value, (int, float)) and 'label' in key.lower():
                label = int(value)
        if parts:
            return (' '.join(parts), label)

        return None

    def extract_features_and_label(self, example: Dict, domain: str) -> Optional[tuple]:
        """
        WHAT: Extract features and labels from dataset example
        WHY:  Different datasets have different schemas
        HOW:  Schema-aware text extraction → text-to-feature encoding
        """
        try:
            result = self._extract_text_and_label(example, domain)
            if result is None:
                return None

            text, label = result
            if not text or not text.strip():
                return None

            # Truncate very long texts to keep encoding fast
            if len(text) > 2000:
                text = text[:2000]

            features = self.encode_text(text, self.num_inputs)
            return (features, label)
        except Exception:
            return None

    def extract_with_metadata(self, example: Dict, domain: str) -> Optional[Dict]:
        """
        WHAT: Extract features, label, and metadata from dataset example
        WHY:  Socratic training needs metadata (domain, raw text) for
              active learning stages (explanation grading, research, etc.)
        HOW:  Same extraction as extract_features_and_label + raw text + domain
        """
        try:
            result = self._extract_text_and_label(example, domain)
            if result is None:
                return None

            text, label = result
            if not text or not text.strip():
                return None

            if len(text) > 2000:
                text = text[:2000]

            features = self.encode_text(text, self.num_inputs)
            return {
                "features": features,
                "label": label,
                "domain": domain,
                "text": text[:500],  # Keep truncated copy for grading
            }
        except Exception:
            return None

    def train_on_batch(self, batch: List[tuple], progress: TrainingProgress,
                       socratic=None, domain: str = "general"):
        """
        WHAT: Train brain on a batch of examples
        WHY:  Incremental learning from streaming data
        HOW:  Extract features, call brain.learn() (or socratic.train_example())
        """
        if not batch:
            return

        print(f"    Training on batch of {len(batch)} examples...", end='', flush=True)

        start_time = time.time()
        trained_count = 0
        total_loss = 0.0

        first_error_logged = False
        for features, label in batch:
            try:
                if socratic:
                    result = socratic.train_example(features, str(label), domain)
                    loss = result.get("loss")
                else:
                    loss = self.brain.learn(features, str(label))
                if loss is not None:
                    total_loss += float(loss)
                trained_count += 1
            except Exception as e:
                if not first_error_logged:
                    print(f"\n    [learn error: {e}, features len={len(features)}]", end='', flush=True)
                    first_error_logged = True

        elapsed = time.time() - start_time
        examples_per_sec = trained_count / max(elapsed, 0.001)
        avg_loss = total_loss / max(trained_count, 1)

        print(f" done ({trained_count}/{len(batch)}, loss={avg_loss:.4f}, {examples_per_sec:.1f} ex/sec)")

        progress.examples_processed += trained_count
        progress.batches_completed += 1

    def cleanup_temp_files(self):
        """
        WHAT: Clean up temporary files after batch processing
        WHY:  Free disk space for next batch
        HOW:  Delete temp directory contents
        """
        if self.config.temp_dir.exists():
            try:
                shutil.rmtree(self.config.temp_dir)
                self.config.temp_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"  ⚠ Warning: Failed to cleanup temp files: {e}")

    def process_dataset_streaming(self, dataset_config: Dict) -> TrainingProgress:
        """
        WHAT: Main streaming processing loop for one dataset
        WHY:  Handle massive datasets efficiently
        HOW:  Stream → Buffer → Train → Cleanup → Repeat
        """
        name = dataset_config['name']
        domain = dataset_config['domain']

        print(f"\n{'='*70}")
        print(f"Processing: {name} ({domain})")
        print(f"{'='*70}")

        # Load or create progress tracker
        progress = TrainingProgress.load(name, self.config.checkpoint_dir)

        # Skip if already completed
        if progress.examples_processed > 0 and self.config.max_examples_per_dataset:
            if progress.examples_processed >= self.config.max_examples_per_dataset:
                print(f"  ✓ Already completed ({progress.examples_processed} examples)")
                return progress

        examples_since_checkpoint = 0

        try:
            # Load dataset — returns IterableDataset, NOT a generator
            dataset = self.load_streaming_dataset(dataset_config)
            if dataset is None:
                elapsed = time.time() - progress.start_time
                print(f"  ✓ Completed: {progress.examples_processed} examples in {elapsed:.1f}s")
                return progress

            # Use iter() + next() directly — avoids generator yield issues
            # with NIMCP C library exception handling
            stream = iter(dataset)
            stream_pos = 0
            logged_keys = False

            while True:
                # Phase 1: Buffer one batch from the stream
                batch = []
                stream_exhausted = False
                while len(batch) < self.config.batch_size:
                    try:
                        example = next(stream)
                        stream_pos += 1
                    except StopIteration:
                        stream_exhausted = True
                        break

                    if not logged_keys:
                        print(f"  📋 Fields: {list(example.keys())}", flush=True)
                        logged_keys = True

                    # Skip already processed examples (if resuming)
                    if stream_pos - 1 < progress.examples_processed:
                        continue

                    result = self.extract_features_and_label(example, domain)
                    if result is not None:
                        batch.append(result)

                    # Progress indicator
                    if stream_pos % 10000 == 0:
                        elapsed = time.time() - progress.start_time
                        rate = progress.examples_processed / max(elapsed, 1)
                        print(f"  📊 Streamed: {stream_pos}, Trained: {progress.examples_processed} ({rate:.1f} ex/sec)", flush=True)

                # Phase 2: Train on the batch
                if batch:
                    self.train_on_batch(batch, progress)
                    examples_since_checkpoint += len(batch)

                    # Checkpoint periodically
                    if examples_since_checkpoint >= self.config.checkpoint_interval:
                        progress.save(self.config.checkpoint_dir)
                        print(f"  💾 Checkpoint at {progress.examples_processed} examples")
                        examples_since_checkpoint = 0
                        gc.collect()

                # Check if stream is done
                if stream_exhausted:
                    break

                # Check limit
                if self.config.max_examples_per_dataset:
                    if progress.examples_processed >= self.config.max_examples_per_dataset:
                        print(f"  ⏹ Reached limit of {self.config.max_examples_per_dataset} examples")
                        break

            # Final checkpoint
            progress.save(self.config.checkpoint_dir)

            elapsed = time.time() - progress.start_time
            print(f"  ✓ Completed: {progress.examples_processed} examples in {elapsed:.1f}s")

        except KeyboardInterrupt:
            print(f"\n  ⏸ Interrupted - saving checkpoint...")
            progress.save(self.config.checkpoint_dir)
            raise

        except Exception as e:
            print(f"  ✗ Error: {e}")
            progress.save(self.config.checkpoint_dir)

        # Final cleanup
        self.cleanup_temp_files()

        return progress


def main():
    """Main streaming training pipeline"""

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='NIMCP Streaming Training Pipeline')
    parser.add_argument('--brain-path', type=str, default=None, help='Path to brain binary to load')
    parser.add_argument('--output-path', type=str, default=None, help='Path to save trained brain')
    parser.add_argument('--batch-size', type=int, default=1000, help='Examples per training batch')
    parser.add_argument('--max-examples', type=int, default=None, help='Max examples per dataset (default: unlimited)')
    parser.add_argument('--checkpoint-interval', type=int, default=10000, help='Checkpoint every N examples')
    parser.add_argument('--datasets', type=str, default=None, help='Comma-separated dataset names (default: all)')
    parser.add_argument('--hf-token', type=str, default=None, help='HuggingFace API token for gated datasets')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    args = parser.parse_args()

    # Load dataset config
    config_file = Path(__file__).parent / "foundation_datasets_config.json"
    with open(config_file) as f:
        config = json.load(f)

    # Initialize streaming config
    stream_config = StreamConfig(
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        max_examples_per_dataset=args.max_examples
    )

    print(f"""
{'='*70}
NIMCP Streaming Training Pipeline
{'='*70}
Configuration:
  Brain Path: {args.brain_path or '(create new)'}
  Output Path: {args.output_path or '(none)'}
  Batch Size: {stream_config.batch_size} examples
  Checkpoint Interval: {stream_config.checkpoint_interval} examples
  Max Examples per Dataset: {stream_config.max_examples_per_dataset or 'unlimited'}
{'='*70}
""")

    # Load brain from frontend or create new
    nimcp.init()
    if args.brain_path and Path(args.brain_path).exists():
        print(f"Loading brain from {args.brain_path}...")
        brain = nimcp.Brain.load(args.brain_path)
        print("Brain loaded successfully")
    else:
        print("Creating new brain...")
        brain = nimcp.Brain("streaming", 2, 2, 128, 32)
        print("Brain created")

    # Determine brain's num_inputs for feature encoding
    num_inputs = 128
    try:
        ni = brain.num_inputs
        if ni and ni > 0:
            num_inputs = ni
    except (AttributeError, Exception):
        pass
    print(f"Feature vector size: {num_inputs}")

    # Create processor
    hf_token = args.hf_token or os.environ.get('HF_TOKEN')
    processor = StreamingDatasetProcessor(brain, stream_config, num_inputs=num_inputs, hf_token=hf_token)

    # Filter datasets if specified
    hf_datasets = [d for d in config['datasets'] if d['type'] == 'huggingface']
    if args.datasets:
        dataset_names = set(args.datasets.split(','))
        hf_datasets = [d for d in hf_datasets if d['name'] in dataset_names]

    print(f"Processing {len(hf_datasets)} datasets in streaming mode\n")

    # Process each dataset
    results = {}
    total_examples = 0

    try:
        for dataset_config in hf_datasets:
            name = dataset_config['name']
            progress = processor.process_dataset_streaming(dataset_config)
            results[name] = progress
            total_examples += progress.examples_processed

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    # Save trained brain
    if args.output_path:
        print(f"Saving trained brain to {args.output_path}...")
        try:
            Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
            brain.save(args.output_path)
            print("Brain saved successfully")
        except Exception as exc:
            print(f"Failed to save brain: {exc}")

    # Summary
    print(f"""
{'='*70}
Training Summary
{'='*70}
Total Examples Processed: {total_examples}
Datasets Completed: {len(results)}

Per-Dataset Results:
""")

    for name, progress in results.items():
        print(f"  {name}: {progress.examples_processed} examples, {progress.batches_completed} batches")

    # Skip nimcp.shutdown() — it triggers a segfault during cleanup.
    # The OS reclaims all memory when the process exits anyway.
    # The brain object must be deleted before exit to flush any pending writes.
    del brain
    gc.collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
