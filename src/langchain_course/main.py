from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openrouter import ChatOpenRouter
from langchain_ollama import ChatOllama


load_dotenv()

def main() -> None:
    print("Hello from langchain-course!")
    information = """
    Lamine Yamal Nasraoui Ebana[A][B] (born 13 July 2007) is a Spanish professional footballer who plays as a right winger for the La Liga club Barcelona and the Spain national team. He is widely regarded as one of the best players in the world.[C]

Yamal was a member of Barcelona's youth academy, La Masia, before joining the first team in the 2023–24 season at the age of 15. At age 17, he became the youngest player to be nominated for the Ballon d'Or, and in 2024 and 2025 he won the Kopa Trophy, awarded to the best young player in the world. Under manager Hansi Flick, Yamal helped Barcelona win a domestic treble during the 2024–25 season: La Liga, the Copa del Rey and the Supercopa de España. Barcelona retained the La Liga title in the 2025–26 season, with Yamal being named La Liga Player of the Season. At 18 years old, he finished as runner-up for the 2025 Ballon d'Or.

Yamal has represented Spain internationally at various youth levels and made his debut for the senior team in 2023, becoming the youngest player to represent the country at age 16. In 2024, he became the youngest player to score in the UEFA European Championship, earning the Young Player Award as Spain achieved their fourth title.[9] He won a World Cup title with Spain at his FIFA World Cup debut in 2026.

Early life
Lamine Yamal Nasraoui Ebana was born on 13 July 2007, in Esplugues de Llobregat in the Barcelona metropolitan area in Catalonia, Spain.[10][11] Yamal's father, Mounir Nasraoui, is from a village between Larache and Ksar el-Kebir in Morocco.[12] His mother, Sheila Ebana, is from Bata, Equatorial Guinea.[D] Yamal has a younger half-brother and a younger half-sister.[16][17] He has a compound given name to honour two family friends, Lamine and Yamal, who helped his parents financially shortly before Yamal was born.[18] As a baby he was featured in a 2007 UNICEF charity photo shoot with his future club Barcelona, during which he was bathed by a 20-year-old Lionel Messi. The photos went viral online in 2024 after Yamal's father posted one of them online. Two years later, the photos received additional attention in the media as Yamal and Messi prepared to face each other in the 2026 FIFA World Cup final.[E]

An important figure in Yamal's upbringing was his paternal grandmother, Fatima, who migrated from Tangier, Morocco, to Spain in 1990.[22] Facing financial hardship, she worked to anchor the family in Rocafonda, a neighbourhood in the city of Mataró in Catalonia. When Yamal's parents separated during his early childhood, Fatima took on a vital role in raising him, managing his daily routine and ensuring he could attend training sessions.[23] Yamal started playing football at the age of four at local club La Torreta in Granollers, where his mother was living.[24][25] He alternated between Mataró and Granollers to spend time with both his parents.[26] His goal celebrations include a hand gesture indicating the number 304, the last three digits of the Rocafonda postcode 08304. Following his success, his popularisation of the number has been credited with bringing pride and recognition to the working class community of Rocafonda.[27][28]

At six years old, Yamal was scouted by Barcelona and was invited to training sessions at La Masia. He signed for the club in 2014 and moved to Barcelona to live and train at the academy.[14][29] Growing up, Yamal's footballing idols were Ronaldinho, Messi and Neymar.[30]"""



    summary_template = """
    given the information {information} about a person, I want you to create :
    1. A short summary of the person in 2-3 sentences.
    2. A list of 5 interesting facts about the person."""

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    llm = ChatOpenRouter(
        model="gpt-4o-mini",
        temperature=0,
    )
    # llm = ChatOllama(
    #         model="llama3.2:latest",
    #         temperature=0,
    #     )
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response)

if __name__ == "__main__":
    main()