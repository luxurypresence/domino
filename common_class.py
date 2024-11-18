import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# Set up logging configuration
logging.basicConfig(level=logging.INFO)


class SearchMode(Enum):
    """
    Enumeration of different search modes with varying weights
    for different property attributes.
    """
    BALANCED = "balanced"
    VISUAL_FOCUS = "visual_focus"
    FEATURES_FOCUS = "features_focus"
    LOCATION_FOCUS = "location_focus"
    DESCRIPTION_FOCUS = "description_focus"
    BALANCED_WITHOUT_VISUAL = "balanced_without_visual"

@dataclass
class PropertyFilters:
    """
    Dataclass representing filters that can be applied to property search results.
    """
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_bedrooms: Optional[int] = None
    max_bedrooms: Optional[int] = None
    min_bathrooms: Optional[int] = None
    max_bathrooms: Optional[int] = None
    property_type: Optional[str] = None
    must_have_amenities: List[str] = field(default_factory=list)
    sale_lease: str = ''