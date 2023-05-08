from pydantic import BaseModel


class SplitThemeItem(BaseModel):
    group1: dict
    group2: dict
    theme: str
class MergeThemeItem(BaseModel):
    theme1: str
    theme2: str
    new_theme: str


class ThemeName(BaseModel):
    theme: str

class RenameTheme(BaseModel):
    theme: str
    new_name: str