"""The European Macro Analyst: the euro-area and Italian backdrop for the instrument.

Italian equities are priced off the ECB's rate path, euro-area inflation and the
BTP-Bund spread as much as off their own results: banks (ISP, UCG) earn on the
rate level and hold BTPs, utilities and infrastructure (ENEL, TRN, SRG) are
priced like bonds, and the spread moves the whole FTSE MIB. This analyst reads
that backdrop from ECB and Eurostat data and says what it means for the
instrument, so the researchers argue over numbers rather than headlines.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.turn import take_turn
from tradingagents.agents.context import get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.tools import get_euro_macro_indicators, get_global_news

# The tools this analyst is offered; its tool node is built from the same tuple.
TOOLS = (
    get_euro_macro_indicators,
    get_global_news,
)


def create_euro_macro_analyst(llm):
    def euro_macro_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        asset_label = "company" if asset_type == "stock" else "asset"
        instrument_context = get_instrument_context_from_state(state)

        system_message = (
            "You are a European macro analyst covering the euro area and Italy. Write a report on the "
            f"macroeconomic backdrop and what it implies for the {asset_label} under analysis. "
            "Ground every claim in data from get_euro_macro_indicators(indicator, curr_date, look_back_days), "
            "which serves ECB and Eurostat series. Cover, at minimum: "
            "(1) monetary policy: 'ecb_deposit_rate' and 'euribor_3m', and whether the market path is easing, on hold or tightening; "
            "(2) inflation: 'euro_hicp', 'euro_core_hicp', 'italy_hicp', and how far they sit from the ECB's 2% target; "
            "(3) Italian sovereign risk: 'btp_bund_spread' (in basis points), 'italy_10y' and 'germany_10y'; "
            "(4) the Italian cycle: 'italy_gdp_growth', 'italy_industrial_production', 'italy_unemployment', 'italy_economic_sentiment'; "
            "(5) the currency: 'eur_usd', where it matters for an exporter or for dollar revenues. "
            "Use get_global_news(curr_date, look_back_days, limit) for ECB communication, Italian fiscal news "
            "and rating actions that the data does not yet show. "
            f"Then map the backdrop to the {asset_label}: name the channels through which it reaches this "
            "instrument (net interest margin and BTP holdings for banks, discount rates for utilities and "
            "regulated infrastructure, financing costs for leveraged companies, demand for cyclicals, FX for "
            "exporters) and say which way each one points. Separate what the data shows from your inference, "
            "and give each figure with its period. A tool result marked DATA_UNAVAILABLE or showing no "
            "observations means the figure is unknown: say so and do not estimate it."
            " Make sure to append a Markdown table at the end of the report listing each indicator, its "
            "latest value and period, its direction, and its implication for the instrument."
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " Report what your tools support; another agent decides the trade."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in TOOLS]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        result, report = take_turn(prompt, llm, TOOLS, state["messages"])

        return {
            "messages": [result],
            "euro_macro_report": report,
        }

    return euro_macro_analyst_node
