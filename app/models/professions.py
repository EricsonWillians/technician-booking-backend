# app/models/professions.py

from enum import Enum

class ProfessionEnum(str, Enum):
    PLUMBER = "Plumber"
    WELDER = "Welder"
    ELECTRICIAN = "Electrician"
    CARPENTER = "Carpenter"
    MECHANIC = "Mechanic"
    PAINTER = "Painter"
    CHEF = "Chef"
    GARDENER = "Gardener"
    TEACHER = "Teacher"
    DEVELOPER = "Developer"
    NURSE = "Nurse"
    UNKNOWN = "Unknown Profession" 
