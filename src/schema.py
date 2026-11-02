from typing import Literal
from pydantic import BaseModel, Field

Label = Literal[
    "innovator_pharma", "generic_manufacturer", "repackager", "distributor",
    "store_brand", "consumer_health", "homeopathic", "personal_care",
]

TAXONOMY = """- innovator_pharma: develops and markets its own branded prescription drugs
- generic_manufacturer: manufactures generic prescription drugs
- repackager: relabels or repackages drugs made by other companies
- distributor: wholesaler or institutional / medical supply distributor
- store_brand: retailer or retail buying group selling private-label products
- consumer_health: branded OTC / consumer health company
- homeopathic: homeopathic or natural remedy maker
- personal_care: cosmetics or personal care brand with drug-claim products (sunscreen, sanitizer, antiperspirant)"""

EVIDENCE_GUIDE = """How to read the FDA profile:
- share_made_by_other_companies >= 0.3, with mostly generic Rx drugs across many unrelated therapeutic areas,
  means REPACKAGER: repackagers keep the original maker's "Manufactured by" line on the label.
  A generic manufacturer's labels name itself or its own subsidiaries, so this share stays low.
- share_repackaged_text > 0 is strong evidence of a repackager, but its absence does NOT rule one out.
- rx_share_brand_equals_generic high (> 0.7) means generic products (generic manufacturer OR repackager);
  low (< 0.3) means branded drugs (innovator).
- share_homeopathic high means homeopathic.
- share_sunscreen or share_hand_sanitizer high, with OTC products, means personal care
  (or consumer health if the company is a branded OTC medicine house).
- Retailers and retail buying groups mean store brand.
- Distributors and wholesalers often LOOK like store brands or repackagers in FDA data, because they file
  labels for their own-brand products. Use what you know about the company's business, not just the labels.
- Do not decide from words in the company name like "Pharmaceuticals" or "Laboratories".
- Web results vary in quality. Official sources (government registrations, NAICS codes, company websites,
  Bloomberg) outweigh directory or SEO pages, which often contain generic filler."""


class Classification(BaseModel):
    label: Label
    confidence: float = Field(ge=0, le=1, description="How sure you are, from 0 to 1")
    evidence: list[str] = Field(description="2-4 short facts from the profile that support the label")