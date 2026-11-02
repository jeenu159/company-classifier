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


class Classification(BaseModel):
    label: Label
    confidence: float = Field(ge=0, le=1, description="How sure you are, from 0 to 1")
    evidence: list[str] = Field(description="2-4 short facts from the profile that support the label")