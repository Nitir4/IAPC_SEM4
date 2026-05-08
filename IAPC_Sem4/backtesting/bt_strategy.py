# backtesting/bt_strategy.py

import backtrader as bt


class MLSignalStrategy(bt.Strategy):
    """
    Strategy that executes precomputed ML signals.

    Signal Encoding:
        1  -> BUY
       -1  -> SELL
        0  -> HOLD
    """

    params = (
        ('printlog', True),
    )

    def __init__(self):

        # ==================================================
        # Custom data feed lines
        # ==================================================

        self.signal = self.data.signal
        self.confidence = self.data.confidence
        self.risk = self.data.risk

        # ==================================================
        # Order tracking
        # ==================================================

        self.order = None

        # ==================================================
        # Position sizing
        # ==================================================

        # Use 95% of available cash per trade
        self.stake = 0.95

    def log(self, txt, dt=None):

        if self.params.printlog:

            dt = dt or self.datas[0].datetime.date(0)

            print(f'  {dt} | {txt}')

    def notify_order(self, order):

        # ==================================================
        # Waiting states
        # ==================================================

        if order.status in [
            order.Submitted,
            order.Accepted
        ]:

            return

        # ==================================================
        # Order completed
        # ==================================================

        if order.status == order.Completed:

            if order.isbuy():

                self.log(
                    f'BUY EXECUTED | '
                    f'Price={order.executed.price:.2f} | '
                    f'Size={order.executed.size:.0f} | '
                    f'Cost={order.executed.value:.2f}'
                )

            elif order.issell():

                self.log(
                    f'SELL EXECUTED | '
                    f'Price={order.executed.price:.2f} | '
                    f'Size={order.executed.size:.0f} | '
                    f'Value={order.executed.value:.2f}'
                )

        # ==================================================
        # Failed orders
        # ==================================================

        elif order.status in [
            order.Canceled,
            order.Margin,
            order.Rejected
        ]:

            self.log(
                'ORDER FAILED | '
                'Cancelled / Margin / Rejected'
            )

        # ==================================================
        # CRITICAL:
        # Reset pending order tracker
        # ==================================================

        self.order = None

    def next(self):

        # ==================================================
        # DEBUG LOGGING
        # ==================================================

        sig = self.signal[0]

        self.log(
            f'BAR | '
            f'Signal={sig:.0f} | '
            f'Position={self.position.size} | '
            f'Cash={self.broker.getcash():.2f} | '
            f'Close={self.data.close[0]:.2f}'
        )

        # ==================================================
        # Prevent duplicate orders
        # ==================================================

        if self.order:

            self.log(
                'PENDING ORDER EXISTS -> SKIPPING BAR'
            )

            return

        conf = self.confidence[0]
        risk = self.risk[0]

        # ==================================================
        # BUY SIGNAL
        # ==================================================

        if sig == 1:

            # Enter only if flat
            if not self.position:

                cash = self.broker.getcash()

                price = self.data.close[0]

                # Position sizing
                size = int(
                    (cash * self.stake) / price
                )

                self.log(
                    f'BUY ATTEMPT | '
                    f'Conf={conf:.0f} | '
                    f'Risk={risk:.0f} | '
                    f'Price={price:.2f} | '
                    f'Size={size}'
                )

                if size > 0:

                    self.order = self.buy(
                        size=size
                    )

                else:

                    self.log(
                        'BUY SKIPPED | '
                        'Computed size <= 0'
                    )

            else:

                self.log(
                    'BUY SIGNAL IGNORED | '
                    'Already in position'
                )

        # ==================================================
        # SELL SIGNAL
        # ==================================================

        elif sig == -1:

            # Exit only if holding position
            if self.position:

                self.log(
                    f'SELL ATTEMPT | '
                    f'Conf={conf:.0f} | '
                    f'Risk={risk:.0f} | '
                    f'Position={self.position.size}'
                )

                self.order = self.sell(
                    size=self.position.size
                )

            else:

                self.log(
                    'SELL SIGNAL IGNORED | '
                    'No active position'
                )

        # ==================================================
        # HOLD SIGNAL
        # ==================================================

        else:

            self.log(
                'HOLD SIGNAL | No action'
            )