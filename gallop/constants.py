"""Physics and world constants, copied from the reference game (og_game/ballinasloe-gallop.html).

All distances are in pixels, all times in seconds. One tile is T pixels square.
"""

T = 16  # tile size
STEP = 1 / 120  # fixed physics step; the reference game also simulates at 120 Hz

# Tile ids stored in the level grid.
EMPTY, GROUND, STONE, PLANK, WATER, HAY = 0, 1, 2, 3, 4, 5
SOLID_TILES = frozenset({GROUND, STONE, HAY})  # planks are one-way platforms, water kills

# Player
PW, PH = 18, 22  # hitbox
RUN = 145  # top running speed
GRAV = 1350
JUMP = 420  # initial jump velocity
FALLMAX = 560  # terminal fall speed
COYOTE = 0.1  # grace period to jump after running off a ledge
BUFFER = 0.13  # a jump pressed this long before landing still fires
DEAD_T = 0.9  # death animation before respawn (only used when end_on_death=False)

# Ground / air acceleration
ACC_GROUND_IDLE = 1000
ACC_GROUND_RUN = 1300
ACC_GROUND_TURN = 2600
ACC_AIR_IDLE = 250
ACC_AIR_RUN = 950

# Stomping a goose
STOMP_BOUNCE = 300
STOMP_BOUNCE_HELD = 440
STOMP_FREEZE = 0.045  # hit-stop: the world pauses briefly after a stomp

# Goose hitbox
GOOSE_W, GOOSE_H = 16, 14
GOOSE_SPEED = 32
