## Data Readme

> Data for all dealers (not filtered)

[https://obj.umiacs.umd.edu/obj/bucket/finra-trace-v2/view/data_not_filtered](https://obj.umiacs.umd.edu/obj/bucket/finra-trace-v2/view/data_not_filtered)

- dict_dealers_2_dealers_2015_filtered.json

    + Filtered data for analyzing the dealer counterparty network.
    + Format: Json
    
            {
                "dealer_1": {
                    "dealer_23": {
                        "count": 423, # count of the transactions between dealer_1 and dealer_23
                        "volume": 243231312.0, # the total volume of the transactions between dealer_1 and dealer_23
                        "bonds": {  # the bonds that dealer_1 trade with dealer_23
                            "bond_APPL": 234, # the count of transactions that dealer_1 trade with dealer_23 for bond_APPL
                            ...
                        }
                    },
                    ...
                },
                ...
            }

- dict_dealers_trace_2015_filtered.json

    + Filtered data for the dealer and its transaction history
    + Format: Json
    
            {
                "dealer_1": {
                    ["bond_APPL", 134323232.0, "BfC", "2015-03-16"],
                    # [bond_id, volume, trade_type, date], the trade type includes "BfC", "S2C", "BfD", "StD" 
                    #     ("B" stands for buy, "S" stands for sell, "C" stands for client, "D" stands for dealer)
                    ...
                },
                ...
            }
    