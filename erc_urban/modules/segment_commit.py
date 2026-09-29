"""Stop to plan, then retain a gyro-relative heading for bounded travel."""
import math


class SegmentCommit:
    def __init__(self, now, settle_s=2.0, max_m=5.0, max_s=30.0):
        self.settle_s, self.max_m, self.max_s = settle_s, max_m, max_s
        self.reset(now)

    def reset(self, now):
        self.active = False
        self.ready_at = now + self.settle_s
        self.remaining = 0.0
        self.heading = 0.0
        self.started = now

    def begin(self, heading, clearance, margin, now):
        budget = min(self.max_m, max(0.0, clearance - margin))
        if budget < 0.25:
            return False
        self.heading, self.remaining, self.started = heading, budget, now
        self.active = True
        return True

    def update(self, odom, now):
        valid = odom is not None and odom.confidence > 0 and all(
            math.isfinite(x) for x in (odom.dx, odom.dy, odom.dyaw))
        if not valid:
            self.reset(now)
            return "waiting for fresh odometry"
        moved = math.hypot(odom.dx, odom.dy)
        if not self.active:
            if moved > 0.02 or abs(odom.dyaw) > math.radians(2):
                self.ready_at = now + self.settle_s
            return "settling" if now < self.ready_at else "ready"
        self.remaining -= moved
        self.heading = (self.heading - math.degrees(odom.dyaw) + 180) % 360 - 180
        if self.remaining <= 0 or now - self.started >= self.max_s:
            self.reset(now)
            return "segment complete"
        return "committed"
