from minigrid.core.world_object import Wall


class WallExtended(Wall):
    def see_behind(self):
        """Can the agent see behind this object?"""
        return False
