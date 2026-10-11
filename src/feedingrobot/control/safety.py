"""Legacy penetration thresholds checked at every physical substep."""

import copy


class SafetyMonitor:
    def __init__(self, control_config, bean_penetration_limit_m):
        self.config = control_config
        self.bean_limit = float(bean_penetration_limit_m)
        self.reset()

    def reset(self):
        self.penetration_duration_s = 0.
        self.peak_penetration_m = 0.
        self.events = []

    def update(self, applied_contacts, boundary_contacts, dt, time):
        rows = applied_contacts + boundary_contacts
        deepest = max((-row["distance"] for row in rows), default=0.)
        self.peak_penetration_m = max(self.peak_penetration_m, deepest)
        deep = deepest > self.config["penetration_limit_m"]
        self.penetration_duration_s = self.penetration_duration_s + dt if deep else 0.
        bean_deep = any((row.get("bean1_id") or row.get("bean2_id"))
                        and row["distance"] < -self.bean_limit for row in rows)
        confirmed = (deep and self.penetration_duration_s + 1e-12
                     >= self.config["penetration_confirm_s"])
        if bean_deep or confirmed:
            self.events.append(dict(time=float(time), reason="model_penetration",
                penetration_m=deepest, duration_s=self.penetration_duration_s,
                bean_penetration=bool(bean_deep)))
            return "model_penetration"
        return None

    def get_state(self):
        return copy.deepcopy(dict(penetration_duration_s=self.penetration_duration_s,
                                  peak_penetration_m=self.peak_penetration_m, events=self.events))

    def set_state(self, state):
        for name, value in copy.deepcopy(state).items():
            setattr(self, name, value)
