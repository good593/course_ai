import enum 

class CHATBOT_ROLE(enum.Enum):
    user = (enum.auto, "사용자")
    assistant = (enum.auto, "조수(AI)")