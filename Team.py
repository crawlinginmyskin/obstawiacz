class Team:
    def __init__(self, name, stats, venue_stats):
        self.name = name
        self.stats = stats
        self.venue_stats = venue_stats


class TeamStats:
    def __init__(self, wins, draws, losses, goals_for, goals_against):
        self.wins = wins
        self.draws = draws
        self.losses = losses
        self.goals_for = goals_for
        self.goals_against = goals_against

