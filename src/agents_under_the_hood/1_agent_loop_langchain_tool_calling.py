import re

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
from pydantic import ValidationError

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "openai/gpt-4o-mini"

# ------ Tools for the agent to use ------


@tool
def get_product_price(product: str) -> float:
    """
    Get the price of a product in the catalog.
    """
    print(f" >> Executing get_product_price for : {product}")
    prices = {
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard": 89.50,
    }
    return prices.get(product.lower(), 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount to a price and return the final price.
    Available tiers: bronze, silver, gold, platinum.
    """
    print(f" >> Executing apply_discount for : {price} with discount {discount_tier}%")
    discount_percentages = {
        "bronze": 5,
        "silver": 12,
        "gold": 23,
        "platinum": 20,
    }
    discount = discount_percentages.get(discount_tier.lower(), 0)
    return round(price * (1 - discount / 100), 2)


# ------ Agent Loop ------
@traceable(name="Langchain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}

    llm = init_chat_model(
        model=MODEL,
        model_provider="openrouter",
        temperature=0,
        max_tokens=500,
    )
    # A dependent tool becomes available only after a successful price lookup.
    product_price = None
    # A named tier is enough information to apply a discount.
    tier_match = re.search(
        r"\b(bronze|silver|gold|platinum)\b", question, re.IGNORECASE
    )
    discount_tier = tier_match.group(1).lower() if tier_match else None

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant that can answer questions about product prices and apply discounts.\n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. Never guess or assume any product price.\n"
                "2. Only call apply_discount After getting the product price. from get_product_price. Pass the exact price.\n"
                "3. NEVER calculate discounts yourself. Always use the apply_discount tool.\n"
                "A named tier such as gold is sufficient; the tool knows its percentage. "
                "For discount questions, call apply_discount before giving the final price.\n"
                "4. If the user does not specify a discount tier, ask them which tier to use - do NOT assume one.\n"
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n --- Iteration {iteration} ---")
        # Once both inputs are known, no further model decision is needed.
        if product_price is not None and discount_tier is not None:
            final_price = apply_discount.invoke(
                {"price": product_price, "discount_tier": discount_tier}
            )
            answer = (
                f"After applying your {discount_tier} discount, the price goes from "
                f"${product_price:,.2f} to ${final_price:,.2f}. "
                f"You save ${product_price - final_price:,.2f}!"
            )
            print(f" >> Final Answer: {answer}")
            return answer

        available_tools = (
            [get_product_price] if product_price is None else [apply_discount]
        )
        ai_message = llm.bind_tools(available_tools).invoke(messages)
        messages.append(ai_message)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            if not ai_message.content or discount_tier is not None:
                print(" >> Model did not request the required tool; retrying.")
                messages.append(
                    HumanMessage(
                        content="Complete the requested tool calls before answering. "
                        "First get_product_price, then apply_discount with the numeric "
                        "price and the requested tier. A tier such as gold needs no "
                        "additional percentage. Return the final tool result as the answer."
                    )
                )
                continue
            print(f" >> Final Answer: {ai_message.content}")
            return ai_message.content

        # Execute every call inside the loop and return each result to the model.
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f" [Tool Selected] {tool_name} with args {tool_args}")

            try:
                tool_to_use = tools_dict.get(tool_name)
                if tool_to_use is None:
                    raise ValueError(f"Unknown tool: {tool_name}")
                if tool_name == "apply_discount":
                    if product_price is None:
                        raise ValueError("Call get_product_price first.")
                    # Carry the lookup result directly instead of trusting the model to copy it.
                    tool_args = {**tool_args, "price": product_price}
                observation = tool_to_use.invoke(tool_args)
                if tool_name == "get_product_price":
                    product_price = observation
                elif tool_name == "apply_discount":
                    answer = (
                        f"After applying your {tool_args['discount_tier']} discount, "
                        f"the final price is ${observation:,.2f}."
                    )
                    print(f" >> Final Answer: {answer}")
                    return answer
            except (ValidationError, ValueError) as exc:
                observation = f"Tool error: {exc}. Correct the arguments and retry."

            print(f" [Tool Output] {observation}")
            messages.append(
                ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
            )

    raise RuntimeError(f"Agent did not finish within {MAX_ITERATIONS} iterations.")


if __name__ == "__main__":
    print("Starting agent loop...")
    question = "What is the price of a laptop after applying a gold discount?"
    run_agent(question)
