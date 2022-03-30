# Create a midi-to-drum mapping.

# Ignore drum loops with fewer onsets than MIN_NB_ONSETS when not much is happening.
MIN_NB_ONSETS = 5

# Number of notes per drum loop matrix.
NOTES_LENGTH = 32

# Each drum class corresponds to a specified instrument. General MIDI specifications.
DRUM_CLASSES = [
    'Kick',
    'Snare',
    'Hi-hat closed',
    'Hi-hat open',
    'Tom',
    'Tambourine/Timbale',
    'Cymbal',
    'Percussion',
    'Clap',
]

MIDI_DRUM_MAP = {
    36: 0,  # Kick / Bass Drum 1
    38: 1,  # Snare / Electric Snare
    40: 1,  # Snare / Electric Snare
    37: 1,  # Snare / Electric Snare
    48: 5,  # Low Timbale
    50: 6,  # Cymbal
    45: 4,  # Tom  / Low-mid Tom
    47: 5,  # Low Timbale
    43: 4,  # Tom  / Low-mid Tom
    58: 4,  # Tom  / Low-mid Tom
    46: 3,  # Hi-hat Open
    26: 3,  # Hi-hat Open
    42: 2,  # Hi-hat Closed
    22: 2,  # Hi-hat Closed
    44: 2,  # Hi-hat Closed
    49: 7,  # Percussion / Open Hi Conga
    55: 7,  # Percussion / Open Hi Conga
    57: 7,  # Percussion / Open Hi Conga
    52: 7,  # Percussion / Open Hi Conga
    51: 8,  # Clap
    59: 8,  # Clap
    53: 8   # Clap
}

DRUM_MIDI_MAP = [  # piano-roll to MIDI
    36,  # 0 Kick / Bass Drum 1
    40,  # 1 Snare / Electric Snare
    42,  # 2 Hi-hat Closed
    46,  # 3 Hi-hat Open
    47,  # 4 Tom  / Low-mid Tom
    66,  # 5 Low Timbale
    51,  # 6 Cymbal
    63,  # 7 Percussion / Open Hi Conga
    39   # 8 Clap
]

CHOSEN_GENRE = [
    "rock",
    "funk",
    "latin",
    "jazz",
    "hiphop"
]
