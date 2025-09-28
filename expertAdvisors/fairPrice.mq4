//+------------------------------------------------------------------+
//|                                                    fairPrice.mq4 |
//|                        Copyright 2025, Gemini CLI & User |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Gemini CLI & User"
#property link      "https://www.google.com"
#property version   "1.00"
#property strict

//--- EA Description
#property description "Grid Entry Expert Advisor based on distance from a Moving Average."
#property description "Opens a market order when price is far from the MA, then places a grid of pending orders."
#property description "Closes all trades when price returns to the MA."

//--- Include libraries

//+------------------------------------------------------------------+
//| Expert Advisor Input Parameters                                  |
//+------------------------------------------------------------------+
//--- Core Settings
input int      MagicNumber              = 12345; // Unique number to identify trades
input double   Lots                     = 0.01;  // Lot size for all orders
input int      Slippage                 = 3;     // Max slippage in pips

//--- Moving Average Settings
input int      MA_Period                = 200;   // Moving Average Period
input ENUM_MA_METHOD MA_Method          = MODE_SMA; // MA method (Simple, Exponential, etc.)
input ENUM_APPLIED_PRICE MA_Price       = PRICE_CLOSE; // MA price (Close, Open, etc.)

//--- Slow Moving Average for Trend Filter
input bool   UseTrendFilter          = true;    // Enable/Disable the trend filter
input int    Slow_MA_Period          = 800;     // Period for the slow (trend) MA
input ENUM_MA_METHOD Slow_MA_Method  = MODE_SMA;  // Method for the slow MA

//--- Trade Entry Grid Settings
input int      Initial_Trigger_Pips     = 100;   // Min distance from MA to open first trade
input int      NumberOfPendingOrders    = 10;    // Number of pending orders in the grid
input int      PendingOrderRangePips    = 50;    // Pip range to spread pending orders over

//--- Exit and Money Management
input bool     CloseOnMA_Touch          = true;  // Close all trades when price touches the MA
input bool     UseEquityStop            = true;  // Use a hard equity stop loss
input double   EquityStopPercentage     = 5.0;   // Close all trades if drawdown reaches this % of account balance

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   Print("fairPrice EA Initialized.");
   Print("Magic Number: ", MagicNumber);
   Print("Initial Trigger Pips: ", Initial_Trigger_Pips);
   Print("Using Equity Stop: ", UseEquityStop ? "Yes" : "No");
   if(UseEquityStop)
     {
      Print("Equity Stop Percentage: ", EquityStopPercentage, "%");
     }
   Print("Using Trend Filter: ", UseTrendFilter ? "Yes" : "No");
   if(UseTrendFilter)
     {
      Print("Slow MA Period: ", Slow_MA_Period);
     }
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   Print("fairPrice EA Deinitialized. Reason: ", reason);
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- Do not trade if parameters are invalid
   if(Initial_Trigger_Pips <= 0 || NumberOfPendingOrders < 0 || PendingOrderRangePips < 0)
     {
      Print("Error: Invalid input parameters for distances or order counts. Please check EA settings.");
      return;
     }

//--- Calculate Moving Averages
   double fastMA = iMA(Symbol(), 0, MA_Period, 0, MA_Method, MA_Price, 0);
   double slowMA = iMA(Symbol(), 0, Slow_MA_Period, 0, Slow_MA_Method, MA_Price, 0);

//--- Get current prices
   double ask = NormalizeDouble(SymbolInfoDouble(Symbol(), SYMBOL_ASK), _Digits);
   double bid = NormalizeDouble(SymbolInfoDouble(Symbol(), SYMBOL_BID), _Digits);

//--- Calculate distance from MA in pips
   double distancePips;
   if(ask < fastMA)
      distancePips = (fastMA - ask) / (_Point * 10); // Distance for a potential buy
   else
      distancePips = (ask - fastMA) / (_Point * 10); // Distance for a potential sell

//--- Check if any trades for this EA are currently open or pending
   int totalTrades = CountEATrades();

//--- === ENTRY LOGIC ===
   // If no trades are open, check if we should open a new set
   if(totalTrades == 0)
     {
      // Determine the trend direction if the filter is enabled
      bool isUptrend = (!UseTrendFilter || fastMA > slowMA);
      bool isDowntrend = (!UseTrendFilter || fastMA < slowMA);

      // Check for BUY signal (price is far below the MA in an uptrend)
      if(isUptrend && ask < fastMA && distancePips >= Initial_Trigger_Pips)
        {
         OpenInitialTradeAndGrid(OP_BUY, ask, fastMA);
         return; // Exit OnTick after opening trades
        }

      // Check for SELL signal (price is far above the MA in a downtrend)
      if(isDowntrend && ask > fastMA && distancePips >= Initial_Trigger_Pips)
        {
         OpenInitialTradeAndGrid(OP_SELL, bid, fastMA);
         return; // Exit OnTick after opening trades
        }
     }
//--- === EXIT LOGIC ===
   // If trades are open, check if we should close them
   else if(totalTrades > 0 && CloseOnMA_Touch)
     {
      bool closeSignal = false;
      // If there are buy trades open, and price crosses above MA
      if(CountEATrades(OP_BUY) > 0 && bid >= fastMA)
        {
         closeSignal = true;
        }
      // If there are sell trades open, and price crosses below MA
      else if(CountEATrades(OP_SELL) > 0 && ask <= fastMA)
        {
         closeSignal = true;
        }
      
      if(closeSignal)
        {
         Print("Price has returned to the MA. Closing all trades for ", Symbol());
         CloseAllEATrades();
         return; // Exit OnTick after closing trades
        }
     }

//--- === MONEY MANAGEMENT LOGIC ===
   if(UseEquityStop && totalTrades > 0)
     {
      CheckEquityStop();
     }
  }

//+------------------------------------------------------------------+
//| Counts open and pending trades managed by this EA                |
//+------------------------------------------------------------------+
int CountEATrades(int type = -1)
  {
   int count = 0;
   for(int i = OrdersTotal() - 1; i >= 0; i--)
     {
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
        {
         if(OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber)
           {
            if(type == -1 || OrderType() == type)
              {
               count++;
              }
           }
        }
     }
   return count;
  }

//+------------------------------------------------------------------+
//| Placeholder for Opening Trades Function                          |
//+------------------------------------------------------------------+
void OpenInitialTradeAndGrid(int type, double price, double takeProfit)
  {
   //--- Normalize TP to be valid
   double normalizedTP = NormalizeDouble(takeProfit, _Digits);

   //--- Open Initial Market Order
   int ticket = OrderSend(Symbol(), type, Lots, price, Slippage, 0, normalizedTP, "fairPrice Initial", MagicNumber, 0, clrNONE);
   if(ticket < 0)
     {
      Print("Failed to open initial market order. Error: ", GetLastError());
      return; // Stop if the first order fails
     }
   else
     {
      Print("Opened initial market order #", ticket, " at ", price);
     }

   //--- Place Pending Order Grid
   if(NumberOfPendingOrders > 0)
     {
      double stepPips = (double)PendingOrderRangePips / NumberOfPendingOrders;
      double pendingPrice;
      int pendingType;

      for(int i = 1; i <= NumberOfPendingOrders; i++)
        {
         if(type == OP_BUY)
           {
            pendingType = OP_BUYLIMIT;
            pendingPrice = price - (i * stepPips * _Point * 10);
           }
         else // OP_SELL
           {
            pendingType = OP_SELLLIMIT;
            pendingPrice = price + (i * stepPips * _Point * 10);
           }
         
         double normalizedPendingPrice = NormalizeDouble(pendingPrice, _Digits);

         ticket = OrderSend(Symbol(), pendingType, Lots, normalizedPendingPrice, 0, 0, normalizedTP, "fairPrice Grid", MagicNumber, 0, clrNONE);
         if(ticket < 0)
           {
            Print("Failed to place pending order #", i, ". Error: ", GetLastError());
           }
         else
           {
            Print("Placed pending order #", i, " at ", normalizedPendingPrice);
           }
        }
     }
  }

//+------------------------------------------------------------------+
//| Closes all open and pending trades for this EA                   |
//+------------------------------------------------------------------+
void CloseAllEATrades()
  {
   for(int i = OrdersTotal() - 1; i >= 0; i--)
     {
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
        {
         if(OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber)
           {
            int type = OrderType();
            bool result = false;
            string typeStr = "";

            if(type == OP_BUY || type == OP_SELL)
              {
               // It's a market order, close it
               typeStr = (type == OP_BUY) ? "Buy" : "Sell";
               result = OrderClose(OrderTicket(), OrderLots(), OrderClosePrice(), Slippage, clrNONE);
              }
            else if(type == OP_BUYLIMIT || type == OP_SELLLIMIT)
              {
               // It's a pending order, delete it
               typeStr = (type == OP_BUYLIMIT) ? "Buy Limit" : "Sell Limit";
               result = OrderDelete(OrderTicket(), clrNONE);
              }

            if(!result)
              {
               Print("Failed to close/delete ", typeStr, " order #", OrderTicket(), ". Error: ", GetLastError());
              }
            else
              {
               Print("Successfully closed/deleted ", typeStr, " order #", OrderTicket());
              }
           }
        }
     }
  }


//+------------------------------------------------------------------+
//| Placeholder for Equity Stop Function                             |
//+------------------------------------------------------------------+
void CheckEquityStop()
  {
   if(EquityStopPercentage <= 0) return; // Do nothing if disabled

   double totalProfit = 0;
   int openTrades = 0;

   //--- Calculate total profit of open trades for this EA
   for(int i = OrdersTotal() - 1; i >= 0; i--)
     {
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
        {
         if(OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber)
           {
            if(OrderType() == OP_BUY || OrderType() == OP_SELL)
              {
               totalProfit += OrderProfit() + OrderSwap() + OrderCommission();
               openTrades++;
              }
           }
        }
     }

   //--- Check only if there are open trades and the profit is negative (drawdown)
   if(openTrades > 0 && totalProfit < 0)
     {
      double accountBalance = AccountBalance();
      double drawdownPercent = (MathAbs(totalProfit) / accountBalance) * 100;

      if(drawdownPercent >= EquityStopPercentage)
        {
         Print("Equity Stop Loss triggered! Drawdown: ", NormalizeDouble(drawdownPercent, 2), "%. Closing all trades.");
         CloseAllEATrades();
        }
     }
  }
//+------------------------------------------------------------------+
