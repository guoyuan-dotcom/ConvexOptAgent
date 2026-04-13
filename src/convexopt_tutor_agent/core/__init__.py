__all__ = ["AppSettings", "SettingsStore", "TutorWorkflowService", "UserFacingError"]


def __getattr__(name: str):
    if name == "AppSettings":
        from convexopt_tutor_agent.core.schema import AppSettings

        return AppSettings
    if name == "SettingsStore":
        from convexopt_tutor_agent.core.settings_store import SettingsStore

        return SettingsStore
    if name == "TutorWorkflowService":
        from convexopt_tutor_agent.core.workflow import TutorWorkflowService

        return TutorWorkflowService
    if name == "UserFacingError":
        from convexopt_tutor_agent.core.workflow import UserFacingError

        return UserFacingError
    raise AttributeError(name)
