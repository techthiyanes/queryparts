from enum import Enum
from typing import Dict, List, Optional

from fuzzywuzzy import fuzz  # For fuzzy matching
from pydantic import BaseModel, Field, field_validator


class Color(str, Enum):
    RED = "red"
    BLUE = "blue"
    GREEN = "green"
    # Add more colors as needed

class FacetType(str, Enum):
    COLOR = "color"
    SIZE = "size"
    MATERIAL = "material"
    # Add more types as needed

class Facet(BaseModel):
    name: str
    value: str
    type: FacetType

    @field_validator('value')
    @classmethod
    def validate_value(cls, v, info):
        if 'type' in info.data:
            if info.data['type'] == FacetType.COLOR:
                try:
                    Color(v.lower())
                except ValueError:
                    raise ValueError(f"{v} is not a valid color")
        return v

class Category(BaseModel):
    name: str
    subcategories: List['Category'] = Field(default_factory=list)
    facets: List[Facet] = Field(default_factory=list)

class AlternativeInterpretation(BaseModel):
    categories: List[Category]
    facets: List[Facet]
    confidence: float

class DecomposedQuery(BaseModel):
    original_query: str
    categories: List[Category]
    facets: List[Facet]
    recommended_facets: List[Facet] = Field(default_factory=list)
    confidence: float = Field(default=1.0)
    alternative_interpretations: List[AlternativeInterpretation] = Field(default_factory=list)

class CategorySchema(BaseModel):
    name: str
    subcategories: Optional[List['CategorySchema']] = None
    allowed_facets: Optional[List[FacetType]] = None

class QueryDecomposerConfig(BaseModel):
    category_schema: List[CategorySchema]

class QueryDecomposer:
    def __init__(self, llm_backend, config: QueryDecomposerConfig):
        self.llm = llm_backend
        self.config = config

    def decompose(self, query: str, context: Optional[Dict] = None) -> DecomposedQuery:
        # Use the LLM to decompose the query into categories and facets
        decomposed = self._llm_decompose(query, context)
        
        # Process categories and facets based on the config
        categories = self._process_categories(decomposed['categories'], decomposed['facets'])
        
        # Resolve conflicts between user-defined and LLM-recommended facets
        facets = self._resolve_facet_conflicts(decomposed['facets'], decomposed['recommended_facets'])
        
        return DecomposedQuery(
            original_query=query,
            categories=categories,
            facets=facets,
            recommended_facets=[Facet(name=k, value=v, type=t) for k, v, t in decomposed['recommended_facets']],
            confidence=decomposed['confidence'],
            alternative_interpretations=[
                AlternativeInterpretation(
                    categories=self._process_categories(alt['categories'], alt['facets']),
                    facets=[Facet(name=k, value=v, type=t) for k, v, t in alt['facets']],
                    confidence=alt['confidence']
                ) for alt in decomposed['alternative_interpretations']
            ]
        )

    def _llm_decompose(self, query: str, context: Optional[Dict]) -> Dict:
        # Placeholder for LLM interaction
        # In a real implementation, this would call the LLM and parse its output
        return {
            'categories': ['clothing', 'shirts'],
            'facets': [('color', 'red', FacetType.COLOR), ('neck-type', 'polo', FacetType.SIZE)],
            'recommended_facets': [('material', 'cotton', FacetType.MATERIAL), ('brand', 'unknown', FacetType.MATERIAL)],
            'confidence': 0.9,
            'alternative_interpretations': [
                {
                    'categories': ['clothing', 'tops'],
                    'facets': [('color', 'red', FacetType.COLOR), ('style', 'polo', FacetType.SIZE)],
                    'confidence': 0.7
                }
            ]
        }

    def _process_categories(self, category_list: List[str], facets: List[tuple]) -> List[Category]:
        # Implementation remains similar to the previous version
        pass

    def _resolve_facet_conflicts(self, user_facets: List[tuple], recommended_facets: List[tuple]) -> List[Facet]:
        user_facet_dict = {f[0]: Facet(name=f[0], value=f[1], type=f[2]) for f in user_facets}
        for name, value, facet_type in recommended_facets:
            if name not in user_facet_dict:
                user_facet_dict[name] = Facet(name=name, value=value, type=facet_type)
        return list(user_facet_dict.values())

    def filter(self, decomposed_query: DecomposedQuery, category: str = None, facets: Dict[str, str] = None) -> DecomposedQuery:
        # Implement filtering logic with exact matching
        filtered_categories = decomposed_query.categories
        if category:
            filtered_categories = [c for c in decomposed_query.categories if c.name == category]
        
        filtered_facets = decomposed_query.facets
        if facets:
            filtered_facets = [f for f in decomposed_query.facets if f.name in facets and f.value == facets[f.name]]
        
        return DecomposedQuery(
            original_query=decomposed_query.original_query,
            categories=filtered_categories,
            facets=filtered_facets,
            recommended_facets=decomposed_query.recommended_facets,
            confidence=decomposed_query.confidence,
            alternative_interpretations=decomposed_query.alternative_interpretations
        )

    def search(self, decomposed_queries: List[DecomposedQuery], search_term: str, threshold: int = 80) -> List[DecomposedQuery]:
        # Implement searching logic with fuzzy matching
        results = []
        for query in decomposed_queries:
            if (fuzz.partial_ratio(search_term.lower(), query.original_query.lower()) >= threshold or
                any(fuzz.partial_ratio(search_term.lower(), cat.name.lower()) >= threshold for cat in query.categories) or
                any(fuzz.partial_ratio(search_term.lower(), f"{f.name}: {f.value}".lower()) >= threshold for f in query.facets)):
                results.append(query)
        return results

# Example usage
schema = QueryDecomposerConfig(
    category_schema=[
        CategorySchema(
            name="clothing",
            subcategories=[
                CategorySchema(
                    name="shirts",
                    allowed_facets=[FacetType.COLOR, FacetType.SIZE]
                )
            ],
            allowed_facets=[FacetType.MATERIAL]
        )
    ]
)

decomposer = QueryDecomposer(llm_backend=None, config=schema)  # Replace None with actual LLM backend
result = decomposer.decompose("red polo shirt")
print(result.json(indent=2))

# Example of filtering
filtered_result = decomposer.filter(result, category="shirts", facets={"color": "red"})
print(filtered_result.json(indent=2))

# Example of searching
search_results = decomposer.search([result], "blue tshirt")
for search_result in search_results:
    print(search_result.json(indent=2))