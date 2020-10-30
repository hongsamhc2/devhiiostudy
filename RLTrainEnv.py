

class RLtradingTestEnv:
    def __init__(self,df=None):
        self.df = df
        self.row_index = 0

    def reset(self):
        self.row_index = 0

    def observe(self):
        if self.df is not None and self.row_index + 1 < len(self.df):
            close = self.df['close'].iloc[self.row_index]
            date = self.df['date'].iloc[self.row_index]
            obs= [date,close]
            self.row_index += 1
            return obs
        return None







