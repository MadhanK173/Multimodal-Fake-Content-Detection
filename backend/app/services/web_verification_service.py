import os

import re
import logging
from html import unescape
from html.parser import HTMLParser

from pathlib import Path

from typing import Any, Dict, List, Optional, Set, Tuple

from urllib.parse import urlparse



import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv





# ============================================================

# PROJECT ENVIRONMENT

# ============================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parents[3] if len(CURRENT_FILE.parents) >= 4 else Path.cwd()

ENV_FILE = PROJECT_ROOT / "backend" / ".env"

load_dotenv(ENV_FILE)



TAVILY_SEARCH_URL = "https://api.tavily.com/search"


logger = logging.getLogger(__name__)


class _ReadablePageTextParser(HTMLParser):
    """Extract readable text from an HTML source page."""

    SKIP_TAGS = {
        "script",
        "style",
        "noscript",
        "svg",
        "nav",
        "footer",
        "header",
        "form",
        "aside",
        "template",
        "iframe",
        "head",
    }

    BLOCK_TAGS = {
        "p",
        "div",
        "section",
        "article",
        "main",
        "li",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "blockquote",
        "pre",
        "table",
        "tr",
        "br",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.skip_depth = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: List[Tuple[str, Optional[str]]],
    ) -> None:
        tag = tag.lower()

        if self.skip_depth > 0:
            if tag in self.SKIP_TAGS:
                self.skip_depth += 1
            return

        if tag in self.SKIP_TAGS:
            self.skip_depth = 1
            return

        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(
        self,
        tag: str,
        attrs: List[Tuple[str, Optional[str]]],
    ) -> None:
        if self.skip_depth == 0 and tag.lower() in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if self.skip_depth > 0:
            if tag in self.SKIP_TAGS:
                self.skip_depth -= 1
            return

        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth == 0 and data:
            self.parts.append(data)

    def get_text(self) -> str:
        text = unescape(" ".join(self.parts))
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n", text)
        return text.strip()


class WebVerificationService:

    """

    Live verification for text and audio transcripts only.



    IMPORTANT UI CONTRACT

    ---------------------

    The web service may inspect Tavily search-result text internally and

    may open the single selected source URL to extract claim-relevant

    factual evidence from the actual webpage. It NEVER returns the full

    article body/raw page content to the client.



    The client receives only:

        - web verdict

        - small status message

        - clickable source title + URL for sources that directly support

          the submitted claim



    Images are intentionally NOT handled here.

    """



    METRIC_PATTERNS: List[Tuple[str, Tuple[str, ...]]] = [

        ("population", ("population", "people", "inhabitants", "residents", "population size")),

        ("revenue", ("revenue", "sales", "turnover", "income")),

        ("profit", ("profit", "net profit", "earnings")),

        ("box_office", ("box office", "grossed", "gross", "collection", "collections")),

        ("price", ("price", "cost", "worth", "valued")),

        ("percentage", ("percent", "%", "rate", "share")),

        ("casualties", ("deaths", "killed", "casualties", "fatalities")),

        ("distance", ("km", "kilometres", "kilometers", "miles", "distance")),

        ("temperature", ("temperature", "degrees", "°c", "°f")),

        ("speed", ("speed", "km/h", "mph")),

        ("date", ("date", "on ", "in 19", "in 20")),

    ]



    GENERIC_ENTITY_WORDS: Set[str] = {

        "the", "this", "that", "these", "those", "a", "an", "according",

        "today", "yesterday", "tomorrow", "news", "report", "reports",

        "article", "articles", "source", "sources", "official", "officials",

        "people", "government", "company", "companies", "country", "countries",

    }



    STOPWORDS: Set[str] = {

        "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for",

        "with", "by", "from", "at", "as", "is", "are", "was", "were", "be",

        "been", "being", "this", "that", "these", "those", "it", "its", "their",

        "they", "he", "she", "we", "you", "has", "have", "had", "will", "would",

        "can", "could", "should", "may", "might", "according", "said", "says",

        "reported", "reportedly", "news", "claim", "claimed", "claims",

    }



    SUPPORT_PHRASES: Tuple[str, ...] = (

        "according to", "confirmed", "confirmation", "officially", "official statement",

        "successfully", "was reported", "has reported", "records show", "data show",

        "data shows", "research shows", "research found", "documented", "announced",

        "revealed", "states that", "shows that", "found that", "reported that",

    )



    CONTRADICTION_PHRASES: Tuple[str, ...] = (

        "false", "incorrect", "wrong", "factually incorrect", "not true", "untrue",

        "did not", "didn't", "never happened", "no evidence", "debunked",

        "misleading", "fabricated", "fake claim", "false claim", "disputed claim",

    )



    def __init__(self) -> None:

        self.api_key = os.getenv("TAVILY_API_KEY", "").strip()

        self.max_results = self._safe_int(

            os.getenv("TAVILY_MAX_RESULTS", "6"), 6, 3, 20

        )

        self.search_depth = os.getenv("TAVILY_SEARCH_DEPTH", "basic").strip().lower()

        if self.search_depth not in {"basic", "advanced"}:

            self.search_depth = "basic"

        self.news_days = self._safe_int(

            os.getenv("TAVILY_NEWS_DAYS", "30"), 30, 1, 365

        )

        self.timeout = 5



    # ============================================================

    # PUBLIC API

    # ============================================================

    def verify_text(

        self,

        text: str,

        model_result: Dict[str, Any],

        modality: str = "text",

    ) -> Dict[str, Any]:

        clean_text = self._clean_text(text)

        if not clean_text:

            raise ValueError("Text for web verification cannot be empty.")



        model_result = self._unwrap_model_result(model_result)

        claim = self._extract_claim(clean_text)



        if not self.api_key:

            return self._not_configured_response(clean_text, model_result, modality, claim)



        queries = self._build_search_queries(claim)

        news_query = self._build_news_query(claim)

        raw_results: List[Dict[str, Any]] = []

        failed: List[str] = []



        for query in queries[:1]:

            try:

                raw_results.extend(self._search_tavily(query, "general"))

            except Exception as exc:

                print(f"Tavily search failed: {exc}")

                failed.append(query)



        try:

            raw_results.extend(

                self._search_tavily(

                    news_query,

                    "news",

                    self.news_days,

                )

            )

        except Exception as exc:

            print(f"Tavily news search failed: {exc}")

            failed.append(news_query)



        unique_results = self._deduplicate_results(raw_results)

        analyzed = [self._analyze_result(claim, r) for r in unique_results]



        # Only results that DIRECTLY support the exact claim are surfaced

        # as clickable "Related Sources" to the frontend.

        supporting = [

            r for r in analyzed

            if r["evidence_type"] == "supports_exact_claim"

        ]



        raw_contradicting = [

            r for r in analyzed

            if r["evidence_type"] == "contradicts_claim"

        ]



        # Contradicting evidence must be strong enough to override a claim.

        # A weak/ambiguous single result is not treated as definitive false

        # evidence. Strong authoritative contradictions can qualify alone;

        # otherwise require at least two strong independent contradictions.

        strong_contradicting = [

            r for r in raw_contradicting

            if self._is_strong_contradiction(r)

        ]



        if len(strong_contradicting) >= 2:

            contradicting = strong_contradicting

        elif (

            len(strong_contradicting) == 1

            and self._is_authoritative_source(

                strong_contradicting[0].get("domain", "")

            )

        ):

            contradicting = strong_contradicting

        else:

            contradicting = []



        relevant = [

            r for r in analyzed

            if r["relevance_score"] >= 0.30

        ]

        related_only = [

            r for r in relevant

            if r["evidence_type"] == "related_not_confirming"

        ]



        # Keep additional relevant live results even when they are not

        # strong enough to be classified as supporting or contradicting.

        # This is important for claims such as government announcements,

        # where related news may exist without directly confirming the

        # exact wording of the submitted claim.

        related_candidates = list(related_only)

        used_urls = {

            str(r.get("url", "")).rstrip("/").lower()

            for r in supporting + contradicting

            if r.get("url")

        }



        for candidate in sorted(

            relevant,

            key=lambda item: float(item.get("relevance_score", 0.0)),

            reverse=True,

        ):

            evidence_type = candidate.get("evidence_type")

            url_key = str(candidate.get("url", "")).rstrip("/").lower()



            if evidence_type in {

                "supports_exact_claim",

                "contradicts_claim",

            }:

                continue



            if not url_key or url_key in used_urls:

                continue



            related_candidates.append(candidate)

            used_urls.add(url_key)



            if len(related_candidates) >= 6:

                break



        verdict = self._determine_verdict(

            supporting_count=len(supporting),

            conflicting_count=len(contradicting),

            relevant_count=len(relevant),

            total_results=len(unique_results),

        )



        # Minimal source objects: title + URL only.

        # Keep supporting, contradicting, and related live sources.



        supporting_source_links = [

            self._public_source(r)

            for r in supporting[:6]

        ]



        contradicting_source_links = [

            self._public_source(r)

            for r in contradicting[:6]

        ]



        related_source_links = [

            self._public_source(r)

            for r in related_candidates[:6]

        ]



        # ============================================================
        # FINAL SIX-SOURCE CONSIDERATION POOL
        # ============================================================

        candidate_pool = [
            r
            for r in analyzed
            if (
                isinstance(r, dict)
                and str(r.get("url") or "").strip()
                and float(r.get("relevance_score") or 0.0) >= 0.30
            )
        ]

        unique_candidate_pool: List[Dict[str, Any]] = []
        seen_candidate_urls: Set[str] = set()

        for result in sorted(
            candidate_pool,
            key=lambda item: float(
                item.get("relevance_score") or 0.0
            ),
            reverse=True,
        ):
            url_key = str(
                result.get("url") or ""
            ).rstrip("/").lower()

            if not url_key:
                continue

            if url_key in seen_candidate_urls:
                continue

            seen_candidate_urls.add(url_key)
            unique_candidate_pool.append(result)

        final_six_candidates = unique_candidate_pool[:6]

        # The public source list is exactly the same maximum-six pool used
        # for best-source selection.
        all_source_links = [
            self._public_source(result)
            for result in final_six_candidates
            if isinstance(result, dict)
            and str(result.get("url") or "").strip()
        ]

        # ============================================================
        # BEST-SOURCE EVIDENCE INSPECTION
        # ============================================================
        #
        # IMPORTANT CONTRACT:
        # The same best-source strategy is used for REAL and FAKE claims.
        # We first rank the complete six-source pool. The highest-ranked
        # source is inspected first. A source is accepted ONLY when its
        # actual webpage can be retrieved AND a claim-relevant passage can
        # be extracted from that webpage. If either step fails, we continue
        # to the next-ranked source.
        #
        # The final web verdict does not decide which source is inspected.
        # Supporting, contradicting, and relevant sources use the same
        # ranking rule. This keeps the REAL and FAKE evidence behavior
        # identical.
        # ============================================================

        (
            best_evidence_source,
            web_explanation,
            evidence_source_url,
            source_evidence_extracted,
        ) = self._find_best_inspectable_evidence_source(
            claim,
            final_six_candidates,
        )

        if not web_explanation:
            web_explanation = (
                "Live web sources were found, but none of the inspected "
                "source pages contained a sufficiently direct passage "
                "for evidence extraction."
            )

        return {

            "enabled": True,

            "status": "success",

            "verification_modality": modality,

            "search_provider": "Tavily",



            "web_verdict": verdict["verdict"],

            "web_verdict_label": verdict["label"],

            "web_explanation": web_explanation,

            "web_explanation_type": (
                "source_evidence"
                if source_evidence_extracted
                else "source_unavailable"
            ),

            # Exact source whose webpage produced the displayed evidence.
            # Raw webpage content is intentionally never returned.
            "evidence_source_title": (
                str(best_evidence_source.get("title") or "").strip()
                if best_evidence_source
                else ""
            ),

            "evidence_source_url": evidence_source_url,

            "evidence_source_domain": (
                str(best_evidence_source.get("domain") or "").strip()
                if best_evidence_source
                else ""
            ),

            "evidence_source_rank": (
                best_evidence_source.get("_evidence_rank")
                if best_evidence_source
                else None
            ),



            "related_sources_found": len(

                all_source_links

            ),



            "supporting_sources": len(

                supporting_source_links

            ),



            "exact_supporting_sources": len(

                supporting_source_links

            ),



            "supporting_evidence_count": len(

                supporting

            ),



            "contradicting_sources": len(

                contradicting_source_links

            ),



            "contradicting_evidence_count": len(

                contradicting

            ),



            "related_not_confirming_sources": len(

                related_only

            ),



            "total_results_found": len(

                unique_results

            ),



            "relevant_sources_found": len(

                relevant

            ),



            # All live sources

            "sources": all_source_links,



            # Keep evidence types separately

            "supporting_sources_list": supporting_source_links,



            "contradicting_sources_list": contradicting_source_links,



            "related_sources_list": related_source_links,



            "model_prediction": model_result.get(

                "prediction"

            ),



            "model_confidence_percent": model_result.get(

                "confidence_percent"

            ),



            # Backend diagnostics

            "search_queries": queries,

            "news_query": news_query,

            "failed_queries": failed,

            "claim_analysis": claim,

        }



    # ============================================================

    # SEARCH

    # ============================================================

    def _search_tavily(

        self,

        query: str,

        topic: str = "general",

        time_range_days: Optional[int] = None,

    ) -> List[Dict[str, Any]]:

        payload: Dict[str, Any] = {

            "query": query,

            "search_depth": self.search_depth,

            "topic": topic,

            "max_results": self.max_results,

            "include_answer": False,

            "include_raw_content": True,

            "include_images": False,

        }



        if topic == "news":

            payload["time_range"] = self._news_time_range(time_range_days or self.news_days)



        response = requests.post(

            TAVILY_SEARCH_URL,

            json=payload,

            headers={

                "Authorization": f"Bearer {self.api_key}",

                "Content-Type": "application/json",

            },

            timeout=self.timeout,

        )

        response.raise_for_status()

        data = response.json()

        return data.get("results", []) if isinstance(data, dict) else []



    # ============================================================

    # CLAIM UNDERSTANDING

    # ============================================================

    def _extract_claim(self, text: str) -> Dict[str, Any]:

        numbers = self._extract_numbers(text)

        entities = self._extract_entities(text)

        metric = self._detect_metric(text)

        time_reference = self._detect_year(text)

        operator = self._detect_numeric_operator(text)



        lower = text.lower()

        metric_terms = dict(self.METRIC_PATTERNS).get(metric, ())

        predicate_tokens = self._content_tokens(text)

        for token in metric_terms:

            predicate_tokens.discard(token.lower())

        for entity in entities:

            predicate_tokens -= self._content_tokens(entity)



        primary_entity = entities[0] if entities else self._fallback_subject(text)



        return {

            "original_text": text,

            "numbers": numbers,

            "entities": entities,

            "primary_entity": primary_entity,

            "metric": metric,

            "time_reference": time_reference,

            "numeric_operator": operator,

            "predicate_tokens": sorted(predicate_tokens),

            "lower_text": lower,

        }



    def _build_search_queries(self, claim: Dict[str, Any]) -> List[str]:

        original = claim["original_text"]

        entity = claim.get("primary_entity") or ""

        metric = claim.get("metric") or "general"

        number_text = " ".join(n["text"] for n in claim.get("numbers", [])[:2])

        predicate = " ".join(claim.get("predicate_tokens", [])[:10])



        candidates = [

            f'"{original}"',

            f"{entity} {metric} {number_text}".strip(),

            f"{entity} {predicate}".strip(),

        ]

        return self._unique_queries([self._limit_query(q) for q in candidates if q.strip()])



    def _build_news_query(self, claim: Dict[str, Any]) -> str:

        entity = claim.get("primary_entity") or ""

        predicate = " ".join(claim.get("predicate_tokens", [])[:8])

        metric = claim.get("metric") or ""

        text = f"{entity} {predicate} {metric}".strip()

        return self._limit_query(text or claim.get("original_text", ""))



    def _normalize_assertion_text(self, value: str) -> str:

        return re.sub(r"\s+", " ", str(value or "").lower()).strip(" ,:;.-")



    def _assertion_signature(self, text: str) -> List[Dict[str, str]]:

        """

        Extract compact, mutually-comparable factual assertions from a

        sentence/claim. These are intentionally conservative and target

        common fact-checking relations such as location, winner, director,

        capital, and year-specific events.

        """

        value = str(text or "").strip()

        signatures: List[Dict[str, str]] = []



        patterns = [

            (

                "location",

                r"^(?P<subject>.+?)\s+(?:is located in|is situated in|lies in|is in|is based in)\s+(?P<object>[^,.!?]+)",

            ),

            (

                "capital",

                r"^(?:the\s+)?capital of\s+(?P<subject>.+?)\s+is\s+(?P<object>[^,.!?]+)",

            ),

            (

                "capital_reverse",

                r"^(?P<object>[^,.!?]+?)\s+is\s+(?:the\s+)?capital of\s+(?P<subject>[^,.!?]+)",

            ),

            (

                "director",

                r"^(?P<subject>.+?)\s+was directed by\s+(?P<object>[^,.!?]+)",

            ),

            (

                "winner",

                r"^(?P<subject>.+?)\s+won\s+(?P<object>.+?)(?:\s+in\s+(?:19|20)\d{2})?$",

            ),

            (

                "introduced_year",

                r"^(?P<subject>.+?)\s+(?:was\s+)?(?:introduced|launched|released)\s+(?:by\s+(?P<object>[^,.!?]+?)\s+)?in\s+(?P<year>(?:19|20)\d{2})",

            ),

            (

                "event_year",

                r"^(?P<subject>.+?)\s+(?:occurred|happened|fell|ended|began|started|landed|declared|became independent)\s+.*?\bin\s+(?P<year>(?:19|20)\d{2})",

            ),

            (

                "largest",

                r"^(?P<subject>.+?)\s+is\s+(?:the\s+)?(?P<object>largest|smallest)\s+(.+)$",

            ),

            (

                "named_relation",

                r"^(?P<subject>.+?)\s+is\s+(?P<object>the\s+highest|the\s+lowest|the\s+first|the\s+last)\s+(.+)$",

            ),

        ]



        normalized = self._normalize_assertion_text(value)



        for kind, pattern in patterns:

            match = re.search(pattern, normalized, flags=re.IGNORECASE)

            if not match:

                continue



            normalized_kind = "capital" if kind == "capital_reverse" else kind



            data = {

                "kind": normalized_kind,

                "subject": self._normalize_assertion_text(

                    match.groupdict().get("subject", "")

                ),

                "object": self._normalize_assertion_text(

                    match.groupdict().get("object", "")

                ),

                "year": self._normalize_assertion_text(

                    match.groupdict().get("year", "")

                ),

            }



            if data["subject"] or data["object"] or data["year"]:

                signatures.append(data)



        return signatures



    def _has_structured_assertion(self, claim: Dict[str, Any]) -> bool:

        """

        Return True when the claim contains a structured factual relation.

        For these claims, topic/keyword overlap alone is not enough to call a

        result exact support.

        """

        return bool(

            self._assertion_signature(

                claim.get("original_text", "")

            )

        )



    def _compare_assertion_values(self, claim: Dict[str, Any], sentence: str) -> str:

        """

        Compare explicit assertion values between the submitted claim and

        an evidence sentence.



        Returns:

            support      -> sentence states the same assertion

            contradict   -> sentence states a mutually exclusive assertion

            neutral      -> not enough structured information to decide

        """

        claim_signatures = self._assertion_signature(claim.get("original_text", ""))

        source_signatures = self._assertion_signature(sentence)



        if not claim_signatures or not source_signatures:

            return "neutral"



        claim_tokens = self._content_tokens(claim.get("original_text", ""))



        for claim_sig in claim_signatures:

            for source_sig in source_signatures:

                if claim_sig["kind"] != source_sig["kind"]:

                    continue



                claim_subject = self._content_tokens(claim_sig["subject"])

                source_subject = self._content_tokens(source_sig["subject"])



                if claim_subject and source_subject:

                    subject_overlap = len(claim_subject & source_subject) / max(1, len(claim_subject))

                    if subject_overlap < 0.60:

                        continue



                # Year-specific relations: same event + same year supports;

                # a different year in the same event context contradicts.

                if claim_sig["year"]:

                    if source_sig["year"]:

                        if claim_sig["year"] == source_sig["year"]:

                            return "support"

                        return "contradict"



                # Winner/location/director/capital/largest-style relations.

                claim_object = self._content_tokens(claim_sig["object"])

                source_object = self._content_tokens(source_sig["object"])



                if claim_object and source_object:

                    object_overlap = len(claim_object & source_object) / max(1, len(claim_object))

                    if object_overlap >= 0.60:

                        return "support"

                    # Same subject/relation but a materially different object

                    # is a contradiction.

                    if object_overlap == 0:

                        return "contradict"



                # If there is an explicit relation but no useful object,

                # fall back to strong claim-token overlap.

                source_tokens = self._content_tokens(sentence)

                overlap = self._token_overlap(claim_tokens, source_tokens)

                if overlap >= 0.70:

                    return "support"



        return "neutral"



    # ============================================================

    # EVIDENCE ANALYSIS

    # ============================================================

    def _is_authoritative_source(self, domain: str) -> bool:

        domain = str(domain or "").lower().strip()

        authoritative = (

            ".gov",

            ".gov.in",

            ".edu",

            ".ac.in",

            "reuters.com",

            "apnews.com",

            "bbc.com",

            "who.int",

            "un.org",

            "nasa.gov",

            "isro.gov.in",

            "pib.gov.in",

            "eci.gov.in",

            "indiacode.nic.in",

            "esa.int",

        )

        return any(

            domain == item or domain.endswith("." + item)

            for item in authoritative

        )



    def _is_strong_contradiction(self, result: Dict[str, Any]) -> bool:

        """

        A contradiction is strong only when the result has high relevance.

        This prevents a loosely related search result from becoming a

        definitive false signal.

        """

        relevance = float(result.get("relevance_score") or 0.0)



        # The contradiction must already have been identified by the

        # sentence-level assertion/number/year logic. Relevance is then

        # used as an additional confidence gate.

        return relevance >= 0.60



    def _analyze_result(self, claim: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:

        title = str(result.get("title") or "").strip()

        url = str(result.get("url") or "").strip()

        raw_content = str(result.get("raw_content") or "").strip()

        search_content = str(
            result.get("content")
            or result.get("snippet")
            or ""
        ).strip()

        source_content = raw_content or search_content

        body = " ".join(
            value
            for value in (title, source_content)
            if value
        ).strip()

        domain = self._extract_domain(url)



        claim_tokens = self._content_tokens(claim["original_text"])

        body_tokens = self._content_tokens(body)

        overlap = self._token_overlap(claim_tokens, body_tokens)



        entity_match = self._entity_match(claim, body)

        metric_match = self._metric_match(claim["metric"], body)

        relevant_score = 0.0

        relevant_score += 0.45 if entity_match else 0.0

        relevant_score += 0.25 if metric_match else 0.0

        relevant_score += min(0.30, overlap * 0.60)



        # Search result URL/domain credibility is used as a tie-breaker,

        # not as proof by itself.

        credibility = self._domain_credibility(domain)

        if credibility >= 0.9:

            relevant_score += 0.03

        elif credibility < 0:

            relevant_score -= 0.20



        relevant_score = max(0.0, min(1.0, relevant_score))



        evidence_type = "neutral"

        exact = False

        contradiction = False

        evidence_sentence = ""



        # Examine individual sentences so unrelated numbers from an article

        # are never compared against the submitted claim.

        for sentence in self._sentences(body):

            sentence_lower = sentence.lower()

            sentence_tokens = self._content_tokens(sentence)

            sentence_overlap = self._token_overlap(claim_tokens, sentence_tokens)

            sentence_entity = self._entity_match(claim, sentence)

            sentence_metric = self._metric_match(claim["metric"], sentence)



            if not sentence_entity:

                continue



            # For quantitative claims, the metric must be present in the

            # same sentence. This prevents unrelated numbers such as

            # millionaire counts, bitcoin supply, foreign-born totals, etc.

            # from being compared with a population claim merely because

            # the article also mentions the entity.

            if claim.get("numbers") and claim.get("metric") != "general_fact":

                if not sentence_metric:

                    continue

                strong_context = True

            else:

                strong_context = sentence_metric or sentence_overlap >= 0.35



            if not strong_context:

                continue



            if self._is_hypothetical(sentence_lower):

                continue



            # First compare structured factual assertions such as

            # "located in X", "won Y", "directed by Z", "capital is X",

            # and year-specific events. This prevents a sentence mentioning

            # the same topic but a different factual value from being treated

            # as supporting evidence.

            assertion_relation = self._compare_assertion_values(

                claim,

                sentence,

            )



            if assertion_relation == "support":

                exact = True
                evidence_sentence = sentence.strip()

                break



            if assertion_relation == "contradict":

                contradiction = True
                evidence_sentence = sentence.strip()

                break



            numeric_relation = self._compare_sentence_numbers(claim, sentence)

            time_relation = self._compare_sentence_year(claim, sentence)



            if numeric_relation == "support" or time_relation == "support":

                exact = True
                evidence_sentence = sentence.strip()

                break



            if numeric_relation == "contradict" or time_relation == "contradict":

                contradiction = True
                evidence_sentence = sentence.strip()

                break



            # For unstructured/general claims, lexical overlap can help.

            # For structured claims (location, capital, winner, director,

            # year-specific relations, etc.), do NOT promote a source to exact

            # support merely because it talks about the same topic.

            #

            # This is the key false-positive guard: a source saying

            # "Jupiter is the largest planet" must not support a claim saying

            # "Mars is the largest planet".

            if not self._has_structured_assertion(claim):

                if not claim.get("numbers") and not claim.get("time_reference"):

                    if (

                        not self._has_contradiction_language(sentence_lower)

                        and (

                            self._has_support_language(sentence_lower)

                            or sentence_overlap >= 0.58

                        )

                    ):

                        exact = True
                        evidence_sentence = sentence.strip()

                        break



            # Contradiction language is only accepted when the sentence is

            # strongly tied to the submitted claim. This prevents a generic

            # fact-checking phrase such as "false claim" elsewhere in an

            # article from creating a contradiction.

            if (

                self._has_contradiction_language(sentence_lower)

                and sentence_overlap >= 0.55

                and sentence_entity

            ):

                contradiction = True



        if exact:

            evidence_type = "supports_exact_claim"

            relevant_score = max(relevant_score, 0.60)

        elif contradiction:

            evidence_type = "contradicts_claim"

            relevant_score = max(relevant_score, 0.60)

        elif relevant_score >= 0.30:

            evidence_type = "related_not_confirming"



        return {

    # INTERNAL fields are consumed by the verifier only.

    "title": title,

    "url": url,

    "domain": domain,

    "relevance_score": round(relevant_score, 3),

    "evidence_type": evidence_type,

    "source_type": str(result.get("published_date") or ""),

    "_source_content": source_content,

    "_evidence_sentence": evidence_sentence,

}



    def _number_belongs_to_metric(self, metric: str, sentence: str, number_text: str) -> bool:

        metric = str(metric or "general_fact")

        if metric == "general_fact":

            return True



        lower = sentence.lower()

        number_text = str(number_text or "").strip().lower()

        pos = lower.find(number_text)

        if pos < 0:

            return False



        left = lower[max(0, pos - 90):pos]

        right = lower[pos:min(len(lower), pos + 70)]



        if metric == "population":

            # 1) Explicit population statement: population ... number.

            pop_pos = left.rfind("population")

            if pop_pos >= 0:

                clause = left[pop_pos + len("population"):].strip(" ,:-")

                # Reject downstream actions such as "population is so large

                # that they could send 2 million...".

                action_words = (

                    "send", "sent", "could send", "can send", "earn",

                    "earned", "own", "owns", "hold", "holds", "use",

                    "uses", "receive", "received", "export", "imports",

                )

                if not any(action in clause for action in action_words):

                    if re.search(r"(?:\bis\b|\bwas\b|\bare\b|\bwere\b|\bof\b|\bat\b|\bestimated\b|\baround\b|\babout\b|\bapproximately\b|\broughly\b)", clause):

                        return True



            # 2) Direct constructions such as "India has 50 million people".

            if re.search(

                rf"\b(?:has|have|had)\s+(?:a\s+)?{re.escape(number_text)}\s+(?:people|inhabitants|residents)\b",

                left + " " + right,

            ):

                return True



            # 3) Explicit "X people live in..." / "X residents in...".

            if re.search(

                rf"\b{re.escape(number_text)}\s+(?:people|inhabitants|residents)\s+(?:live|lives|inhabit|reside)\b",

                right,

            ):

                return True



            return False



        aliases = {

            "revenue": ("revenue", "sales", "turnover", "income"),

            "profit": ("profit", "earnings", "net income"),

            "box_office": ("box office", "grossed", "collection", "collections", "gross"),

            "price": ("price", "cost", "worth", "valued"),

            "percentage": ("%", "percent", "percentage", "rate", "share"),

            "casualties": ("deaths", "killed", "casualties", "fatalities"),

            "distance": ("km", "kilometers", "kilometres", "miles", "distance"),

            "temperature": ("temperature", "degrees", "°c", "°f"),

            "speed": ("speed", "km/h", "mph"),

        }

        window = left + " " + right

        return any(alias in window for alias in aliases.get(metric, (metric,)))



    def _compare_sentence_numbers(self, claim: Dict[str, Any], sentence: str) -> str:

        claim_numbers = claim.get("numbers", [])

        if not claim_numbers:

            return "neutral"



        source_numbers = self._extract_numbers(sentence)

        if not source_numbers:

            return "neutral"



        operator = claim.get("numeric_operator", "exact")

        for c in claim_numbers:

            c_base = self._numeric_base(c)

            if c_base is None:

                continue

            for s in source_numbers:

                s_base = self._numeric_base(s)

                if s_base is None:

                    continue

                # Percentage values should not be compared with currency or

                # population values.

                if self._unit_family(c.get("unit", "")) != self._unit_family(s.get("unit", "")):

                    continue



                # The metric itself must describe the source number. This

                # prevents examples such as "India's population is large"

                # followed by "2 million citizens could be sent..." from

                # being interpreted as a 2-million population figure.

                if not self._number_belongs_to_metric(claim.get("metric", "general_fact"), sentence, s.get("text", "")):

                    continue



                ratio = abs(s_base - c_base) / max(abs(c_base), 1.0)



                if operator == "only":

                    # "only 50 million" means the quantity must not be

                    # materially above 50 million. A source stating a

                    # smaller number is compatible but does not prove the

                    # exact wording; still treat it as supporting evidence

                    # for the upper-bound claim.

                    if s_base > c_base * 1.05:

                        return "contradict"

                    if s_base <= c_base * 1.05:

                        return "support"

                if operator in {"at_least", "more_than", "over"}:

                    if s_base >= c_base * 0.95:

                        return "support"

                    return "contradict"

                if operator in {"at_most", "less_than", "under", "below"}:

                    if s_base <= c_base * 1.05:

                        return "support"

                    return "contradict"



                # Numeric claims without an inequality are treated as

                # estimates when the wording does not say "exactly".

                tolerance = 0.05 if self._approximate_claim(claim["original_text"]) else 0.02

                if ratio <= tolerance:

                    return "support"

                if ratio >= 0.08:

                    return "contradict"



        return "neutral"



    def _compare_sentence_year(self, claim: Dict[str, Any], sentence: str) -> str:

        claim_year = claim.get("time_reference")

        if not claim_year:

            return "neutral"



        years = [

            int(x)

            for x in re.findall(r"\b(?:19|20)\d{2}\b", sentence)

        ]

        if not years:

            return "neutral"



        lower = sentence.lower()

        predicate = claim.get("predicate_tokens", [])

        claim_predicate_count = len(predicate)

        predicate_hits = sum(

            1

            for token in predicate

            if token in lower

        )



        # A year is strong evidence only when the sentence also expresses

        # the same factual relation. A different year somewhere in a source

        # is NOT enough to contradict the claim.

        #

        # Example:

        #   Claim: France won the FIFA World Cup in 2018.

        #   Source: Argentina won the FIFA World Cup in 2022.

        #

        # The source is related, but it does not establish a contradiction

        # about France's 2018 result. Contradiction therefore requires a

        # matching structured assertion (same subject + relation) below.



        claim_signatures = self._assertion_signature(

            claim.get("original_text", "")

        )

        source_signatures = self._assertion_signature(sentence)



        # Strongest path: compare the same structured relation.

        for claim_sig in claim_signatures:

            if claim_sig.get("year") and claim_sig.get("year") != str(claim_year):

                continue



            for source_sig in source_signatures:

                if claim_sig.get("kind") != source_sig.get("kind"):

                    continue



                claim_subject = self._content_tokens(

                    claim_sig.get("subject", "")

                )

                source_subject = self._content_tokens(

                    source_sig.get("subject", "")

                )



                if claim_subject and source_subject:

                    subject_overlap = (

                        len(claim_subject & source_subject)

                        / max(1, len(claim_subject))

                    )

                    if subject_overlap < 0.75:

                        continue



                # Same relation and same year -> direct support.

                if source_sig.get("year") == str(claim_year):

                    return "support"



                # Same relation and a different year -> contradiction.

                # This is only allowed after the subject/relation match above.

                if source_sig.get("year"):

                    return "contradict"



        # Fallback support: same claim year + meaningful predicate overlap.

        # Never use this fallback to create a contradiction.

        if (

            int(claim_year) in years

            and (

                predicate_hits >= 2

                or (

                    claim_predicate_count > 0

                    and predicate_hits / claim_predicate_count >= 0.50

                )

            )

        ):

            return "support"



        return "neutral"



    # ============================================================

    # VERDICT

    # ============================================================

    def _determine_verdict(

        self,

        supporting_count: int,

        conflicting_count: int,

        relevant_count: int,

        total_results: int,

    ) -> Dict[str, str]:

        if supporting_count >= 1 and conflicting_count == 0:

            return {

                "verdict": "LIKELY_REAL",

                "label": "RELATED SOURCES FOUND",

                "explanation": "Live sources were found that directly support the submitted claim.",

            }



        if conflicting_count >= 1 and supporting_count == 0:

            return {

                "verdict": "LIKELY_FALSE",

                "label": "CONTRADICTING EVIDENCE FOUND",

                "explanation": (

                    "Live sources were found that contradict "

                    "the submitted claim."

                ),

            }



        if supporting_count >= 1 and conflicting_count >= 1:

            return {

                "verdict": "QUESTIONABLE",

                "label": "MIXED WEB EVIDENCE",

                "explanation": "Live sources contain both supporting and conflicting evidence.",

            }



        return {

            "verdict": "UNVERIFIED",

            "label": "NO SUCH RELATED SOURCE FOUND",

            "explanation": "No live source directly supporting the exact claim was found.",

        }

    def _rank_evidence_sources(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Rank the complete source pool from strongest to weakest.

        This is the single ranking used for both REAL and FAKE evidence
        inspection. Direct supporting and direct contradicting evidence are
        treated equally; the final verdict is determined separately.
        """

        if not candidates:
            return []

        scored_candidates: List[Tuple[float, float, Dict[str, Any]]] = []

        for result in candidates:
            if not isinstance(result, dict):
                continue

            evidence_type = str(
                result.get("evidence_type") or ""
            ).strip()

            relevance = float(
                result.get("relevance_score") or 0.0
            )

            domain = str(
                result.get("domain") or ""
            ).strip()

            source_content = str(
                result.get("_source_content") or ""
            ).strip()

            evidence_sentence = str(
                result.get("_evidence_sentence") or ""
            ).strip()

            if evidence_type in {
                "supports_exact_claim",
                "contradicts_claim",
            }:
                evidence_strength = 0.60
            elif evidence_type == "related_not_confirming":
                evidence_strength = 0.20
            else:
                evidence_strength = 0.0

            exact_sentence_bonus = 0.20 if evidence_sentence else 0.0
            content_bonus = 0.10 if len(source_content) >= 80 else 0.0
            authority_bonus = (
                0.08
                if self._is_authoritative_source(domain)
                else 0.0
            )

            selection_score = (
                evidence_strength
                + (relevance * 0.30)
                + exact_sentence_bonus
                + content_bonus
                + authority_bonus
            )

            scored_candidates.append(
                (
                    selection_score,
                    relevance,
                    result,
                )
            )

        scored_candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
            ),
            reverse=True,
        )

        return [item[2] for item in scored_candidates]

    def _select_best_evidence_source(
        self,
        candidates: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Return the highest-ranked source without opening its URL.

        Kept for compatibility with any existing internal caller. Actual
        evidence generation must use ``_find_best_inspectable_evidence_source``
        so that a source is not accepted merely because its URL responds.
        """

        ranked = self._rank_evidence_sources(candidates)
        return ranked[0] if ranked else None

    def _find_best_inspectable_evidence_source(
        self,
        claim: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], str, str, bool]:
        """
        Inspect the sources strictly from best to worst and return the first
        source that produces real evidence from its actual webpage.

        A source is accepted only when BOTH conditions are satisfied:

        1. Its actual URL can be retrieved successfully.
        2. The retrieved webpage contains a claim-relevant passage that
           ``_extract_source_evidence`` can extract.

        If the strongest source is blocked, empty, dynamically unusable, or
        contains no relevant passage, the next-ranked source is inspected.
        This behavior is identical for REAL and FAKE claims.
        """

        if not candidates:
            return None, "", "", False

        ranked_candidates = self._rank_evidence_sources(candidates)

        for rank, result in enumerate(ranked_candidates[:4], start=1):
            source_url = str(
                result.get("url") or ""
            ).strip()

            source_title = str(
                result.get("title") or ""
            ).strip()

            if not source_url:
                continue

            logger.info(
                "Evidence inspection candidate #%s: %s (%s)",
                rank,
                source_title or source_url,
                source_url,
            )

            source_content = self._fetch_source_page_content(
                source_url
            )

            if not source_content:
                logger.info(
                    "Evidence inspection skipped candidate #%s because "
                    "the webpage could not be retrieved: %s",
                    rank,
                    source_url,
                )
                continue

            evidence = self._extract_source_evidence(
                claim,
                source_content,
            )

            if not evidence:
                logger.info(
                    "Evidence inspection skipped candidate #%s because "
                    "no claim-relevant passage was found on the webpage: %s",
                    rank,
                    source_url,
                )
                continue

            # Raw webpage content stays server-side. Only the extracted
            # evidence and public source metadata are returned.
            result["_source_page_content"] = source_content
            result["_selected_evidence"] = evidence
            result["_evidence_rank"] = rank

            logger.info(
                "Selected evidence source #%s: %s (%s)",
                rank,
                source_title or source_url,
                source_url,
            )

            return result, evidence, source_url, True

        return None, "", "", False

    # ============================================================
    # ACTUAL SOURCE PAGE CONTENT
    # ============================================================

    def _fetch_source_page_content(
        self,
        url: str,
    ) -> str:
        """
        Open the selected source URL and extract readable text from
        the actual webpage. The extracted content remains internal.
        """

        url = str(url or "").strip()

        if not url:
            return ""

        try:
            response = requests.get(
                url,
                timeout=3,
                allow_redirects=True,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 "
                        "(Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 "
                        "(KHTML, like Gecko) "
                        "Chrome/154.0 Safari/537.36"
                    ),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )

            response.raise_for_status()

            content_type = str(
                response.headers.get("Content-Type") or ""
            ).lower()

            if content_type and not any(
                value in content_type
                for value in (
                    "text/html",
                    "application/xhtml+xml",
                    "text/plain",
                )
            ):
                return ""

            html = response.text

            if not html:
                return ""

            parser = _ReadablePageTextParser()
            parser.feed(html)
            parser.close()

            text = parser.get_text()
            text = re.sub(r"[ \t\r\f\v]+", " ", text)
            text = re.sub(r"\n\s*\n+", "\n", text).strip()

            # Keep an upper bound so a very large page cannot overwhelm
            # the evidence matching stage.
            return text[:250000]

        except Exception as exc:
            logger.warning(
                "Failed to fetch selected source page %s: %s",
                url,
                exc,
            )
            return ""


    # ============================================================
    # ACTUAL SOURCE EVIDENCE
    # ============================================================

    def _extract_source_evidence(
        self,
        claim: Dict[str, Any],
        source_content: str,
    ) -> str:
        """
        Extract the most relevant factual passage directly from the
        selected source webpage. No claim restatement is generated.
        """

        source_content = str(source_content or "").strip()
        source_content = re.sub(
            r"[ \t\r\f\v]+",
            " ",
            source_content,
        )
        source_content = re.sub(
            r"\n\s*\n+",
            "\n",
            source_content,
        ).strip()

        if not source_content:
            return ""

        claim_text = str(
            claim.get("original_text") or ""
        ).strip()

        if not claim_text:
            return ""

        claim_tokens = self._content_tokens(claim_text)

        if not claim_tokens:
            return ""

        sentences = self._sentences(source_content)

        if not sentences:
            return ""

        scored_sentences: List[Tuple[float, int, str]] = []

        for index, sentence in enumerate(sentences):
            sentence = sentence.strip()

            if not sentence:
                continue

            sentence_lower = sentence.lower()
            sentence_tokens = self._content_tokens(sentence)

            if not sentence_tokens:
                continue

            overlap = self._token_overlap(
                claim_tokens,
                sentence_tokens,
            )

            entity_match = self._entity_match(
                claim,
                sentence,
            )

            metric_match = self._metric_match(
                claim.get("metric", "general_fact"),
                sentence,
            )

            assertion_relation = self._compare_assertion_values(
                claim,
                sentence,
            )

            numeric_relation = self._compare_sentence_numbers(
                claim,
                sentence,
            )

            time_relation = self._compare_sentence_year(
                claim,
                sentence,
            )

            score = 0.0

            if entity_match:
                score += 0.45

            if metric_match:
                score += 0.20

            score += min(0.25, overlap * 0.50)

            if assertion_relation in {"support", "contradict"}:
                score += 0.65
            elif numeric_relation in {"support", "contradict"}:
                score += 0.55
            elif time_relation in {"support", "contradict"}:
                score += 0.45

            if self._is_hypothetical(sentence_lower):
                score -= 0.50

            noise_phrases = (
                "accept cookies",
                "cookie policy",
                "privacy policy",
                "terms of use",
                "subscribe to our newsletter",
                "sign up for our newsletter",
            )

            if any(phrase in sentence_lower for phrase in noise_phrases):
                continue

            if 8 <= len(sentence.split()) <= 100:
                score += 0.05

            if score <= 0.25:
                continue

            scored_sentences.append(
                (
                    score,
                    index,
                    sentence,
                )
            )

        if not scored_sentences:
            return ""

        scored_sentences.sort(
            key=lambda item: (
                item[0],
                -item[1],
            ),
            reverse=True,
        )

        _, best_index, best_sentence = scored_sentences[0]
        selected_sentences = [best_sentence]

        if best_index + 1 < len(sentences):
            next_sentence = sentences[best_index + 1].strip()

            if next_sentence:
                next_tokens = self._content_tokens(next_sentence)
                next_overlap = self._token_overlap(
                    claim_tokens,
                    next_tokens,
                )

                next_entity = self._entity_match(
                    claim,
                    next_sentence,
                )

                if next_overlap >= 0.20 or next_entity:
                    selected_sentences.append(next_sentence)

        evidence = " ".join(selected_sentences).strip()
        words = evidence.split()

        if len(words) > 90:
            evidence = " ".join(words[:90]).rstrip(".,;:") + "..."

        return evidence


    # ============================================================
    # CLAIM-SPECIFIC EXPLANATION

    # ============================================================

    def _build_claim_specific_explanation(

        self,

        *,

        clean_text: str,

        model_result: Dict[str, Any],

        supporting_count: int,

        contradicting_count: int,

        related_count: int,

        relevant_count: int,

    ) -> str:

        """

        Build a short, sentence-form explanation tied directly to the

        submitted claim and the evidence state returned by live search.

        """



        claim_text = self._clean_text(clean_text).strip()



        # Remove a common news-style prefix so the explanation reads naturally.

        claim_text = re.sub(

            r"^breaking\s*:\s*",

            "",

            claim_text,

            flags=re.IGNORECASE,

        ).strip()



        if len(claim_text) > 420:

            claim_text = claim_text[:417].rstrip() + "..."



        raw_prediction = str(

            model_result.get("prediction")

            or model_result.get("label")

            or ""

        ).strip().lower()



        model_is_fake = (

            raw_prediction in {

                "fake", "false", "label_1", "1", "ai", "synthetic"

            }

            or "fake" in raw_prediction

            or "generated" in raw_prediction

            or "synthetic" in raw_prediction

        )



        if supporting_count > 0 and contradicting_count == 0:

            return (

                f'The submitted claim says: "{claim_text}". '

                "Live sources were found that directly support this claim."

            )



        if contradicting_count > 0 and supporting_count == 0:

            return (

                f'The submitted claim says: "{claim_text}". '

                "Live sources found during verification contain information "

                "that contradicts this claim."

            )



        if supporting_count > 0 and contradicting_count > 0:

            return (

                f'The submitted claim says: "{claim_text}". '

                "The live sources contain both supporting and contradicting "

                "information, so the claim cannot be treated as fully verified."

            )



        if related_count > 0 or relevant_count > 0:

            if model_is_fake:

                return (

                    f'The submitted claim says: "{claim_text}". '

                    "The trained model classified the claim as fake, while the "

                    "live sources checked provide related information but do not "

                    "provide credible confirmation of this specific claim."

                )



            return (

                f'The submitted claim says: "{claim_text}". '

                "The live sources checked provide related information, but not "

                "enough direct evidence to confirm or contradict the exact claim."

            )



        if model_is_fake:

            return (

                f'The submitted claim says: "{claim_text}". '

                "The trained model classified the claim as fake, and the live "

                "search did not provide credible evidence to confirm the specific claim."

            )



        return (

            f'The submitted claim says: "{claim_text}". '

            "The live search did not provide enough direct evidence to confirm "

            "or contradict the exact claim."

        )



    # ============================================================

    # CLAIM HELPERS

    # ============================================================

    def _extract_numbers(self, text: str) -> List[Dict[str, Any]]:

        pattern = re.compile(

            r"(?<![A-Za-z0-9-])"

            r"(?:₹|rs\.?|inr|usd|\\$|€|eur|£|gbp)?\s*"

            r"([+-]?\d+(?:,\d{3})*(?:\.\d+)?)"

            r"\s*"

            r"(crore|cr|lakh|million|billion|trillion|thousand|%|percent|bn|mn|k|m|b)?"

            r"(?![A-Za-z0-9])",

            flags=re.IGNORECASE,

        )

        result: List[Dict[str, Any]] = []

        for match in pattern.finditer(text):

            value = float(match.group(1).replace(",", ""))

            unit = (match.group(2) or "").lower()

            before = text[max(0, match.start() - 24):match.start()].lower()

            after = text[match.end():match.end() + 24].lower()



            # Ignore years/dates/clock values here; years are handled separately.

            if not unit and 1900 <= value <= 2100:

                continue

            if not unit and (re.search(r"\bday\s*$", before) or re.match(r"\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", after)):

                continue

            if not unit and (re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*$", before)):

                continue

            if not unit and (":" in after[:2] or (before.endswith(":") and before[-2:-1].isdigit())):

                continue



            result.append({"value": value, "unit": unit, "text": match.group(0).strip()})

        return result



    def _extract_entities(self, text: str) -> List[str]:

        candidates: List[str] = []



        # Multi/single capitalized proper names, acronyms, and hyphenated names.

        for match in re.finditer(

            r"\b(?:[A-Z][A-Za-z0-9'’.-]*)(?:\s+[A-Z][A-Za-z0-9'’.-]*)*\b",

            text,

        ):

            value = match.group(0).strip()

            if value.lower() in self.GENERIC_ENTITY_WORDS:

                continue

            if value not in candidates and len(value) >= 2:

                candidates.append(value)



        # Important named organizations/places often appear in lowercase text.

        known = (

            "India", "United States", "United Kingdom", "NASA", "ISRO", "ESA",

            "Chandrayaan-3", "SpaceX", "OpenAI", "Microsoft", "Google", "Apple",

            "Tesla", "Samsung", "Netflix", "Toxic", "Tamil Nadu",

        )

        lower = text.lower()

        for item in known:

            if item.lower() in lower and item not in candidates:

                candidates.append(item)



        # Prefer multi-word/proper named entities, then exact known entities.

        candidates.sort(key=lambda x: (-len(self._content_tokens(x)), -len(x)))

        return candidates[:8]



    def _detect_metric(self, text: str) -> str:

        lower = text.lower()



        # Specific factual metrics must win before the generic date matcher.

        # Do NOT use raw substring matching for date phrases such as "on "

        # because words like "million" contain "on " and would make

        # an ordinary population claim look like a date claim.

        specific_order = (

            "population", "revenue", "profit", "box_office",

            "price", "percentage", "casualties", "distance",

            "temperature", "speed",

        )

        pattern_map = dict(self.METRIC_PATTERNS)



        for metric in specific_order:

            for term in pattern_map.get(metric, ()):

                if metric == "percentage" and term == "%":

                    if "%" in lower:

                        return metric

                elif re.search(rf"\b{re.escape(term)}\b", lower):

                    return metric



        # Date is deliberately conservative. Only an actual year/date

        # construction should classify a claim as date-related.

        if re.search(r"\b(?:19|20)\d{2}\b", lower):

            return "date"

        if re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b", lower):

            return "date"

        if re.search(r"\b\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\b", lower):

            return "date"

        return "general_fact"



    def _detect_year(self, text: str) -> Optional[int]:

        match = re.search(r"\b(?:19|20)\d{2}\b", text)

        return int(match.group(0)) if match else None



    def _detect_numeric_operator(self, text: str) -> str:

        lower = text.lower()

        if re.search(r"\bonly\s+[^.]*\d", lower):

            return "only"

        if re.search(r"\b(at least|minimum of|more than|over)\b", lower):

            if "at least" in lower or "minimum of" in lower:

                return "at_least"

            return "more_than"

        if re.search(r"\b(at most|maximum of|less than|under|below)\b", lower):

            if "at most" in lower or "maximum of" in lower:

                return "at_most"

            if "under" in lower:

                return "under"

            return "less_than"

        return "exact"



    def _fallback_subject(self, text: str) -> str:

        metric_pos = len(text)

        for _, terms in self.METRIC_PATTERNS:

            for term in terms:

                p = text.lower().find(term.lower())

                if p >= 0:

                    metric_pos = min(metric_pos, p)

        before = text[:metric_pos].strip(" ,:-")

        tokens = re.findall(r"[A-Za-z][A-Za-z'-]+", before)

        return tokens[-1] if tokens else ""



    # ============================================================

    # SENTENCE / MATCHING HELPERS

    # ============================================================

    def _sentences(self, text: str) -> List[str]:

        # Protect common abbreviations such as "Aug." so a date like

        # "Aug. 23, 2023" remains one evidence sentence.

        protected = str(text or "")

        abbreviations = (

            "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.",

            "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.",

            "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec.",

            "e.g.", "i.e."

        )

        replacements: Dict[str, str] = {}

        for index, abbreviation in enumerate(abbreviations):

            marker = f"__ABBR_{index}__"

            protected = protected.replace(abbreviation, marker)

            replacements[marker] = abbreviation



        parts = re.split(r"(?<=[.!?])\s+|\n+", protected)

        sentences: List[str] = []

        for part in parts:

            value = part.strip()

            if not value:

                continue

            for marker, abbreviation in replacements.items():

                value = value.replace(marker, abbreviation)

            sentences.append(value)

        return sentences



    def _content_tokens(self, value: str) -> Set[str]:

        tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9'-]{2,}\b", value.lower())

        return {t for t in tokens if t not in self.STOPWORDS}



    def _token_overlap(self, left: Set[str], right: Set[str]) -> float:

        if not left or not right:

            return 0.0

        return len(left & right) / max(1, len(left))



    def _entity_match(self, claim: Dict[str, Any], text: str) -> bool:

        lower = text.lower()

        entities = claim.get("entities", [])

        primary = claim.get("primary_entity", "")

        if primary and self._entity_in_text(primary, lower):

            return True

        for entity in entities:

            if self._entity_in_text(entity, lower):

                return True

        return False



    def _entity_in_text(self, entity: str, lower_text: str) -> bool:

        normalized = re.sub(r"[^a-z0-9]+", " ", entity.lower()).strip()

        if not normalized:

            return False

        # Handle variants such as Chandrayaan-3 / Chandrayaan 3.

        pattern = r"\b" + re.escape(normalized).replace(r"\ ", r"\s+") + r"\b"

        if re.search(pattern, re.sub(r"[^a-z0-9]+", " ", lower_text)):

            return True

        return normalized.replace(" ", "") in re.sub(r"[^a-z0-9]+", "", lower_text)



    def _metric_match(self, metric: str, text: str) -> bool:

        if metric == "general_fact":

            return True

        terms = dict(self.METRIC_PATTERNS).get(metric, ())

        lower = text.lower()

        return any(term in lower for term in terms)



    def _numeric_base(self, number: Dict[str, Any]) -> Optional[float]:

        try:

            value = float(number["value"])

        except (KeyError, TypeError, ValueError):

            return None

        unit = str(number.get("unit", "")).lower()

        multipliers = {

            "thousand": 1e3, "k": 1e3,

            "million": 1e6, "mn": 1e6, "m": 1e6,

            "billion": 1e9, "bn": 1e9, "b": 1e9,

            "trillion": 1e12,

            "crore": 1e7, "cr": 1e7,

            "lakh": 1e5,

        }

        if unit in {"%", "percent"}:

            return value

        return value * multipliers.get(unit, 1.0)



    def _unit_family(self, unit: str) -> str:

        unit = unit.lower()

        if unit in {"%", "percent"}:

            return "percentage"

        if unit in {"crore", "cr", "lakh", "million", "mn", "m", "billion", "bn", "b", "trillion", "thousand", "k"}:

            return "quantity"

        if unit in {"usd", "$", "eur", "€", "gbp", "£", "inr", "rs", "₹"}:

            return "currency"

        return "plain"



    def _approximate_claim(self, text: str) -> bool:

        lower = text.lower()

        return any(word in lower for word in ("about", "around", "approximately", "roughly", "nearly", "more than", "over", "almost"))



    def _is_hypothetical(self, sentence: str) -> bool:

        return any(

            phrase in sentence

            for phrase in (

                "what if", "if india had", "hypothetical", "could india have", "imagine if",

                "scenario", "suppose", "what would happen if",

            )

        )



    def _has_support_language(self, sentence: str) -> bool:

        return any(p in sentence for p in self.SUPPORT_PHRASES)



    def _has_contradiction_language(self, sentence: str) -> bool:

        return any(p in sentence for p in self.CONTRADICTION_PHRASES)



    # ============================================================

    # PUBLIC SOURCE OUTPUT

    # ============================================================

    def _public_source(self, result: Dict[str, Any]) -> Dict[str, str]:

        """
        Return the minimal public representation of a Tavily result.

        Tavily source URLs are preserved exactly as returned. The
        verification/router layer performs the final public validation,
        but this method also avoids accidentally dropping the URL when
        Tavily uses one of its normal URL fields.
        """

        if not isinstance(result, dict):
            return {
                "title": "Source",
                "url": "",
            }

        title = (
            result.get("title")
            or result.get("domain")
            or "Source"
        )

        url = (
            result.get("url")
            or result.get("link")
            or result.get("href")
            or ""
        )

        url = str(url).strip()

        # Never manufacture a URL. Preserve only an actual HTTP(S) URL.
        try:
            parsed = urlparse(url)
            if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
                url = ""
        except Exception:
            url = ""

        return {
            "title": str(title).strip()[:300] or "Source",
            "url": url,
        }



    # ============================================================

    # DOMAIN / DEDUP

    # ============================================================

    def _extract_domain(self, url: str) -> str:

        try:

            hostname = urlparse(url).hostname or ""

            return hostname[4:] if hostname.lower().startswith("www.") else hostname.lower()

        except Exception:

            return ""



    def _domain_credibility(self, domain: str) -> float:

        authoritative = (

            ".gov", ".gov.in", ".edu", ".ac.in", "reuters.com", "apnews.com",

            "bbc.com", "who.int", "un.org", "nasa.gov", "isro.gov.in", "pib.gov.in",

            "eci.gov.in", "indiacode.nic.in", "esa.int",

        )

        established = (

            "thehindu.com", "indianexpress.com", "hindustantimes.com", "ndtv.com",

            "news18.com", "deccanherald.com", "theprint.in", "economictimes.indiatimes.com",

            "imdb.com", "forbes.com", "nytimes.com", "washingtonpost.com", "guardian.com",

        )

        social = ("facebook.com", "instagram.com", "x.com", "twitter.com", "tiktok.com")

        if any(domain == d or domain.endswith("." + d) for d in authoritative):

            return 1.0

        if any(domain == d or domain.endswith("." + d) for d in established):

            return 0.8

        if any(domain == d or domain.endswith("." + d) for d in social):

            return -0.2

        return 0.4



    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        output: List[Dict[str, Any]] = []

        seen: Set[str] = set()

        for result in results:

            url = str(result.get("url") or "").strip()

            if not url:

                continue

            key = url.rstrip("/").lower()

            if key in seen:

                continue

            seen.add(key)

            output.append(result)

        return output



    # ============================================================

    # NOT CONFIGURED

    # ============================================================

    def _not_configured_response(

        self,

        text: str,

        model_result: Dict[str, Any],

        modality: str,

        claim: Dict[str, Any],

    ) -> Dict[str, Any]:

        return {

            "enabled": False,

            "status": "not_configured",

            "verification_modality": modality,

            "search_provider": "Tavily",

            "web_verdict": "NOT_CONFIGURED",

            "web_verdict_label": "LIVE VERIFICATION NOT CONFIGURED",

            "web_explanation": "Add TAVILY_API_KEY to backend/.env to enable live verification.",

            "related_sources_found": 0,

            "supporting_sources": 0,

            "exact_supporting_sources": 0,

            "supporting_evidence_count": 0,

            "contradicting_sources": 0,

            "contradicting_evidence_count": 0,

            "related_not_confirming_sources": 0,

            "total_results_found": 0,

            "relevant_sources_found": 0,

            "sources": [],

            "supporting_sources_list": [],

            "contradicting_sources_list": [],

            "related_sources_list": [],

            "model_prediction": model_result.get("prediction"),

            "model_confidence_percent": model_result.get("confidence_percent"),

            "search_queries": [],

            "news_query": "",

            "failed_queries": [],

            "claim_analysis": claim,

        }



    # ============================================================

    # SMALL UTILS

    # ============================================================

    @staticmethod

    def _unwrap_model_result(model_result: Dict[str, Any]) -> Dict[str, Any]:

        if not isinstance(model_result, dict):

            return {}

        if isinstance(model_result.get("result"), dict):

            return model_result["result"]

        if isinstance(model_result.get("classification"), dict):

            return model_result["classification"]

        return model_result



    @staticmethod

    def _clean_text(text: str) -> str:

        return re.sub(r"\s+", " ", str(text or "")).strip()



    @staticmethod

    def _unique_queries(queries: List[str]) -> List[str]:

        output: List[str] = []

        seen: Set[str] = set()

        for query in queries:

            key = query.strip().lower()

            if query.strip() and key not in seen:

                seen.add(key)

                output.append(query.strip())

        return output



    @staticmethod

    def _limit_query(value: str) -> str:

        return re.sub(r"\s+", " ", str(value or "")).strip()[:380]



    @staticmethod

    def _news_time_range(days: int) -> str:

        if days <= 1:

            return "day"

        if days <= 7:

            return "week"

        if days <= 30:

            return "month"

        return "year"



    @staticmethod

    def _safe_int(value: str, default: int, minimum: int, maximum: int) -> int:

        try:

            return max(minimum, min(maximum, int(value)))

        except (TypeError, ValueError):

            return default
