import re

from datetime import datetime, timedelta

from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════
# CATEGORY TAXONOMY
# ═══════════════════════════════════════════════════════════════

CATEGORIES = [
    "Food",
    "Transportation",
    "Shopping",
    "Entertainment",
    "Bills",
    "Healthcare",
    "Education",
    "Travel",
    "Rent",
    "Miscellaneous",
]


KEYWORD_TIERS = {

    "Food": {

        "strong": [
            "restaurant",
            "groceries",
            "grocery",
            "breakfast",
            "lunch",
            "dinner",
            "food delivery",
        ],

        "medium": [
            "pizza",
            "burger",
            "coffee",
            "tea",
            "food",
            "drinks",
            "drink",
        ],

        "weak": [
            "milk",
            "bread",
            "cake",
            "snack",
        ],
    },


    "Transportation": {

        "strong": [
            "uber",
            "ola",
            "taxi",
            "cab",
            "fuel",
            "petrol",
        ],

        "medium": [
            "bus",
            "train",
            "ticket",
            "ride",
            "transport",
        ],

        "weak": [
            "parking",
            "toll",
        ],
    },


    "Shopping": {

        "strong": [
            "amazon",
            "flipkart",
            "myntra",
            "shopping",
        ],

        "medium": [
            "clothes",
            "shoes",
            "dress",
            "bag",
            "watch",
            "purchase",
            "bought",
            "order",
            "laptop",
            "mobile",
        ],

        "weak": [
            "electronics",
            "gadget",
        ],
    },


    "Entertainment": {

        "strong": [
            "netflix",
            "spotify",
            "movie ticket",
        ],

        "medium": [
            "movie",
            "game",
            "concert",
            "music",
            "subscription",
        ],

        "weak": [
            "fun",
            "play",
        ],
    },


    "Bills": {

        "strong": [
            "electricity bill",
            "internet bill",
            "wifi bill",
            "mobile bill",
        ],

        "medium": [
            "bill",
            "recharge",
            "electricity",
            "internet",
            "insurance",
        ],

        "weak": [
            "payment",
            "invoice",
        ],
    },


    "Healthcare": {

        "strong": [
            "hospital",
            "clinic",
            "doctor",
            "pharmacy",
        ],

        "medium": [
            "medicine",
            "health",
            "dentist",
            "checkup",
        ],

        "weak": [
            "vitamins",
            "fitness",
        ],
    },


    "Education": {

        "strong": [
            "school fee",
            "college fee",
            "course fee",
            "udemy",
            "coursera",
        ],

        "medium": [
            "books",
            "course",
            "training",
            "education",
        ],

        "weak": [
            "study",
            "exam",
        ],
    },


    "Travel": {

        "strong": [
            "flight ticket",
            "hotel booking",
            "airbnb",
        ],

        "medium": [
            "trip",
            "vacation",
            "travel",
            "hotel",
        ],

        "weak": [
            "luggage",
        ],
    },


    "Rent": {

        "strong": [
            "house rent",
            "flat rent",
            "room rent",
        ],

        "medium": [
            "rent",
            "rental",
            "lease",
        ],

        "weak": [
            "deposit",
        ],
    },


    "Miscellaneous": {
        "strong": [],
        "medium": [],
        "weak": [],
    }
}


# ═══════════════════════════════════════════════════════════════
# MERCHANT MAP
# ═══════════════════════════════════════════════════════════════

MERCHANT_MAP = {

    "swiggy": "Food",
    "zomato": "Food",
    "dominos": "Food",
    "starbucks": "Food",

    "uber": "Transportation",
    "ola": "Transportation",
    "rapido": "Transportation",

    "amazon": "Shopping",
    "flipkart": "Shopping",
    "myntra": "Shopping",

    "netflix": "Entertainment",
    "spotify": "Entertainment",

    "jio": "Bills",
    "airtel": "Bills",

    "apollo": "Healthcare",

    "udemy": "Education",
    "coursera": "Education",

    "oyo": "Travel",
}


_TIER_WEIGHTS = {
    "strong": 3,
    "medium": 2,
    "weak": 1,
}


# ═══════════════════════════════════════════════════════════════
# AMOUNT EXTRACTION
# ═══════════════════════════════════════════════════════════════

_AMOUNT_RE = re.compile(
    r'(?P<sym>[₹$€£])?\s*(?P<amt>\d[\d,]*(?:\.\d{1,2})?)',
    re.IGNORECASE,
)


def extract_amount(text):

    matches = list(
        _AMOUNT_RE.finditer(text)
    )

    if not matches:

        return 0.0, "INR"

    best = max(

        matches,

        key=lambda m:
            float(
                m.group("amt")
                .replace(",", "")
            )
    )

    amount = float(
        best.group("amt")
        .replace(",", "")
    )

    return amount, "INR"


# ═══════════════════════════════════════════════════════════════
# CATEGORY SCORING
# ═══════════════════════════════════════════════════════════════

def score_categories(text):

    lower = text.lower()

    scores = {
        cat: 0
        for cat in CATEGORIES
    }


    # MERCHANT BOOST

    for merchant, category in MERCHANT_MAP.items():

        if merchant in lower:

            scores[category] += 10


    # KEYWORD MATCHING

    for category, tiers in KEYWORD_TIERS.items():

        for tier, keywords in tiers.items():

            weight = _TIER_WEIGHTS[tier]

            for kw in keywords:

                if kw in lower:

                    scores[category] += weight

    return scores


def resolve_category(scores):

    best_category = max(
        scores,
        key=scores.get
    )

    best_score = scores[best_category]

    total = sum(scores.values())


    if best_score == 0:

        return "Miscellaneous", 0.0


    confidence = round(
        best_score / total,
        4
    )

    return best_category, confidence


# ═══════════════════════════════════════════════════════════════
# JOINT QUERY SPLITTER
# ═══════════════════════════════════════════════════════════════

def extract_segments(text):

    # SPLIT BY COMMA

    if "," in text:

        segments = [

            s.strip()

            for s in text.split(",")

            if s.strip()
        ]

        return segments


    # SPLIT BY AND

    if " and " in text.lower():

        segments = re.split(
            r'\s+and\s+',
            text,
            flags=re.IGNORECASE
        )

        return [

            s.strip()

            for s in segments

            if s.strip()
        ]


    return [text]


# ═══════════════════════════════════════════════════════════════
# EXPENSE DATA CLASS
# ═══════════════════════════════════════════════════════════════

@dataclass
class ExpenseSegment:

    raw_segment: str

    amount: float

    currency: str

    category: str

    confidence: float

    datetime: datetime

    all_scores: dict = field(
        default_factory=dict
    )


    @property
    def confidence_pct(self):

        return (
            f"{self.confidence * 100:.1f}%"
        )


    def to_dict(self):

        return {

            "raw_segment":
                self.raw_segment,

            "amount":
                self.amount,

            "currency":
                self.currency,

            "category":
                self.category,

            "confidence":
                self.confidence_pct,

            "datetime":
                self.datetime.isoformat()
        }


# ═══════════════════════════════════════════════════════════════
# MAIN PARSER CLASS
# ═══════════════════════════════════════════════════════════════

class JointExpenseParser:

    def extract_datetime(self, text):

        text = text.lower().strip()

        now = datetime.now()


        # DEFAULT TIME

        hour = 12
        minute = 0


        # CONTEXTUAL DEFAULTS

        if "morning" in text:

            hour = 9

        elif "afternoon" in text:

            hour = 14

        elif "evening" in text:

            hour = 18

        elif "night" in text:

            hour = 21

        elif "lunch" in text:

            hour = 13

        elif "dinner" in text:

            hour = 20


        # RELATIVE DATE

        base_date = now


        if "yesterday" in text:

            base_date = (
                now - timedelta(days=1)
            )

        elif "tomorrow" in text:

            base_date = (
                now + timedelta(days=1)
            )


        # EXPLICIT TIME
        # Examples:
        # 11 AM
        # 10:30 pm
        # 7pm

        time_match = re.search(

            r'\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b',

            text,

            re.IGNORECASE
        )


        # EXPLICIT TIME MUST OVERRIDE DEFAULTS

        if time_match:

            parsed_hour = int(
                time_match.group(1)
            )

            parsed_minute = int(
                time_match.group(2) or 0
            )

            meridian = (
                time_match.group(3)
                .lower()
            )


            # CONVERT TO 24H FORMAT

            if meridian == "pm":

                if parsed_hour != 12:

                    parsed_hour += 12

            elif meridian == "am":

                if parsed_hour == 12:

                    parsed_hour = 0


            # OVERRIDE DEFAULTS

            hour = parsed_hour
            minute = parsed_minute


        return base_date.replace(

            hour=hour,

            minute=minute,

            second=0,

            microsecond=0
        )


    # MAIN PARSE

    def parse(self, text):

        if not text.strip():

            return []


        results = []


        segments = extract_segments(text)


        for segment in segments:

            amount, currency = (
                extract_amount(segment)
            )

            scores = score_categories(
                segment
            )

            category, confidence = (
                resolve_category(scores)
            )


            results.append(

                ExpenseSegment(

                    raw_segment=segment,

                    amount=amount,

                    currency=currency,

                    category=category,

                    confidence=confidence,

                    datetime=self.extract_datetime(
                        segment
                    ),

                    all_scores=scores,
                )
            )

        return results


    def parse_to_dicts(self, text):

        return [

            s.to_dict()

            for s in self.parse(text)
        ]


    def parse_with_total(self, text):

        segments = self.parse(text)

        total = sum(
            s.amount
            for s in segments
        )

        return segments, total