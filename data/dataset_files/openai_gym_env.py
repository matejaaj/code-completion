import numpy as np
from collections import defaultdict

class Player:
    def __init__(self, name):
        self.name = name
        self.hand = defaultdict(int)
        self.cards_won = defaultdict(int)
        self.points = 0

    def draw_card(self, card):
        self.hand[card] += 1

    def play_card(self, card):
        if self.hand[card] > 0:
            self.hand[card] -= 1
            return card


    def need_cards(self):
        return sum(self.hand.values()) < 4

    def card_count(self):
        return sum(self.hand.values())

    def add_won_cards(self, cards):
        for card, count in cards.items():
            self.cards_won[card] += count
            if card in ['10', 'A']:
                self.points += 10 * count

    def show_hand(self):
        return {card: count for card, count in self.hand.items() if count > 0}

    def show_won_cards(self):
        return {card: count for card, count in self.cards_won.items() if count > 0}


import random
from collections import defaultdict
from player import Player

class CardGame:
    def __init__(self, player_names):
        self.players = [Player(name) for name in player_names]
        self.deck = self.create_deck()
        self.pile = defaultdict(int)
        self.pile_winner_index = 0
        self.first_player_index = 0
        self.current_player_index = 0
        self.first_card_played = None
        self.deal_cards()

    def create_deck(self):
        cards = ['7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        deck = [card for card in cards for _ in range(4)]
        random.shuffle(deck)
        return deck

    def deal_cards(self):
      while any(player.need_cards() for player in self.players) and self.deck:
          for player in self.players:
              if player.need_cards() and self.deck:
                  player.draw_card(self.deck.pop())

    def reset(self):
        self.deck = self.create_deck()
        self.pile = defaultdict(int)
        self.pile_winner_index = 0
        self.first_card_played = None
        self.first_player_index = 0
        self.current_player_index = 0
        for player in self.players:
            player.hand = defaultdict(int)
            player.cards_won = defaultdict(int)
            player.points = 0
        self.deal_cards()

    def step(self, card):
      needs_to_play_again = False
      info = {
        'invalid_move': False,
        'reason': ''
    }
      if card == 'END':
          if self.current_player_index != self.first_player_index:
              reward = -1000
              done = True
              info['invalid_move'] = True
              info['reason'] = 'Played END but not played first'
              return reward, done, needs_to_play_again, info
          elif self.first_card_played is None:
              reward = -1000
              done = True
              info['invalid_move'] = True
              info['reason'] = 'Played END on first turn'
              return reward, done, needs_to_play_again, info
          else:
              reward, done, needs_to_play_again = self._end_round()
              reward += 5
              self.current_player_index = self.pile_winner_index
              return reward, done, needs_to_play_again, info


      if (self.current_player_index == self.first_player_index) and self.first_card_played is not None:
          if card != self.first_card_played and card != '7':
              reward = -1000
              done = True
              info['invalid_move'] = True
              info['reason'] = 'Tried to continue round with invalid card'
              return reward, done, needs_to_play_again, info

      if self.players[self.current_player_index].hand[card] <= 0:
          reward = -1000
          done = True
          info['invalid_move'] = True
          info['reason'] = 'Card not available in hand'
          return reward, done, needs_to_play_again, info

      self.players[self.current_player_index].play_card(card)
      self.pile[card] += 1

      if self.first_card_played is None:
          self.first_card_played = card
          self.pile_winner_index = self.current_player_index
          self.first_player_index = self.current_player_index
      else:
          if card == self.first_card_played or card == '7':
              self.pile_winner_index = self.current_player_index


      if not needs_to_play_again:
        self._next_turn()

      reward = 5
      done = self._check_game_over()
      return reward, done, needs_to_play_again, info

    def _next_turn(self):
        self.current_player_index = (self.current_player_index + 1) % 2

    def _end_round(self):
        self.players[self.pile_winner_index].add_won_cards(self.pile)
        reward = (self.pile['10'] + self.pile['A']) * 10 + 5

        self.pile = defaultdict(int)
        self.first_card_played = None

        needs_to_play_again = False
        if self.first_player_index == self.pile_winner_index:
          needs_to_play_again = True

        self.first_player_index = self.pile_winner_index
        done = self._check_game_over()
        if not done:
          self.deal_cards()

        return reward, done, needs_to_play_again

    def play_opponent_move(self):
        opponent = self.players[self.current_player_index]
        valid_cards = []

        if (self.current_player_index == self.first_player_index) and self.first_card_played is not None:
            valid_cards = [card for card, count in opponent.hand.items() if count > 0 and (card == '7' or card == self.first_card_played)]
            valid_cards.append('END')
        else:
            valid_cards = [card for card in opponent.hand if opponent.hand[card] > 0]

        choice = random.choice(valid_cards)

        _, done, needs_again, _ = self.step(choice)
        return done, needs_again


    def winner_reward(self):
      bonus = 0
      if self._check_game_over():
        if self.players[0].points > self.players[1].points:
          bonus = 50
        if self.players[0].points < self.players[1].points:
          bonus = -30

      return bonus



    def _check_game_over(self):
        if len(self.deck) == 0 and all(player.card_count() == 0 for player in self.players):
            return True
        return False


import gym
from gym import spaces
from collections import defaultdict
import random
from game import CardGame

class PlayerEnv(gym.Env):
    def __init__(self, game_state):
        super(PlayerEnv, self).__init__()
        self.game_state = game_state
        self.action_space = spaces.Discrete(9)  # 8 different cards and one 'END' action
        self.observation_space = spaces.Box(low=0, high=52, shape=(8 + 8 + 8 + 2 + 1,), dtype=int)  # player's hand, pile on the table, won cards, points, first card in pile

    def reset(self):
        self.game_state.reset()
        return self._get_obs()

    def step(self, action):
        card = self._action_to_card(action)
        reward, done, needs_to_play_again, info = self.game_state.step(card)
        if not done and not needs_to_play_again:
            done, needs_again_opponent = self.game_state.play_opponent_move()
            if not done and needs_again_opponent:
                done, _ = self.game_state.play_opponent_move()

        if done:
            reward += self.game_state.winner_reward()

        return self._get_obs(), reward, done, info

    def render(self, mode='human'):
        print(f"\n{'='*20} GAME STATE {'='*20}")

        for i, player in enumerate(self.game_state.players):
            print(f"Player {i+1} ({player.name}):")
            print(f"  Hand: {player.show_hand()}")
            print(f"  Won Cards: {player.show_won_cards()}")
            print(f"  Points: {player.points}")
            print('-' * 50)

        print(f"Pile: {dict(self.game_state.pile)}")
        print(f"First Card Played: {self.game_state.first_card_played}")
        print('='*50)

    def close(self):
        pass

    def _get_obs(self):
        obs = []

        # player's cards
        player = self.game_state.players[0]
        obs.extend([player.hand[card] for card in ['7', '8', '9', '10', 'J', 'Q', 'K', 'A']])

        # pile
        obs.extend([self.game_state.pile[card] for card in ['7', '8', '9', '10', 'J', 'Q', 'K', 'A']])

        # won cards
        combined_won_cards = defaultdict(int)
        for p in self.game_state.players:
            for card, count in p.cards_won.items():
                combined_won_cards[card] += count
        obs.extend([combined_won_cards[card] for card in ['7', '8', '9', '10', 'J', 'Q', 'K', 'A']])

        # points of both players
        for p in self.game_state.players:
            obs.append(p.points)

        # first card in pile
        first_card_index = self._card_to_index(self.game_state.first_card_played)
        obs.append(first_card_index)

        return np.array(obs, dtype=np.int32)

    def _action_to_card(self, action):
        card_map = ['7', '8', '9', '10', 'J', 'Q', 'K', 'A', 'END']
        if action == 8:
            return 'END'
        return card_map[action]

    def _card_to_index(self, card):
        card_map = ['7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        if card is None:
            return -1  # no card on table
        return card_map.index(card)
