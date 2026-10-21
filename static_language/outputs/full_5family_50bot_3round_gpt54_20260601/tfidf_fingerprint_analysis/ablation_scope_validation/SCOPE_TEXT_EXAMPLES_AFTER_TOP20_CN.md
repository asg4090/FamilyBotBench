# 删除 top-20 贡献词后的文本示例

## F1 / f1_original_architecture_001

- 协议：`self_family`
- 词数保留率：`0.886`
- 活跃 TF-IDF 特征保留率：`0.791`
- bot 概率下降：`0.328`
- 实际匹配删除词项：`institutions; governments; investigate; incitement; addiction; questions; civilians; politics; treating; partisan; loyalty; public; crisis; starts; symbol; legal; feels; line; just; prop`

原文节选：

> This is the part that should alarm people beyond this war. When a state starts treating sympathy for civilians, or even dissent from the war line, as suspect speech, the guardrails are already slipping. Maybe some cases involve actual incitement. Fine, investigate that narrowly. But mass arrests over social posts is how democracies teach citizens to self-censor first and ask legal questions later. Bad precedent. Very bad one. This is one of those places where the country keeps failing a very basic test. If there is evidence of crimes or ...

删除后节选：

> This is the part that should alarm people beyond this war. When a state sympathy for , or even dissent from the war , as suspect speech, the guardrails are already slipping. Maybe some cases involve actual . Fine, that narrowly. But mass arrests over social posts is how democracies teach citizens to self-censor first and ask later. Bad precedent. Very bad one. This is one of those places where the country keeps failing a very basic test. If there is evidence of crimes or influence peddling, that. Fine. But ...

## F1 / f1_original_architecture_038

- 协议：`self_family`
- 词数保留率：`0.884`
- 活跃 TF-IDF 特征保留率：`0.753`
- bot 概率下降：`0.330`
- 实际匹配删除词项：`consequences; institutions; politics; treating; violence; question; security; justice; matters; exactly; leaders; quickly; public; caveat; trust; clear; trend; lines; avoid; just`

原文节选：

> This is one of the better trend lines in Asia right now. South Korea and Japan do not have to suddenly trust each other to see that the security math has changed with China, North Korea, and a shakier world economy. Better ties also help the US alliance system, which matters more than people admit. Small caveat though, these resets are fragile, and domestic politics or old historical disputes can still knock them off course pretty quickly. That sounds more dramatic than it is. Mixed reactions usually just mean the ...

删除后节选：

> This is one of the better in Asia right now. South Korea and Japan do not have to suddenly each other to see that the math has changed with China, North Korea, and a shakier world economy. Better ties also help the US alliance system, which more than people admit. Small though, these resets are fragile, and domestic or old historical disputes can still knock them off course pretty . That sounds more dramatic than it is. Mixed reactions usually mean the idea was rolled out poorly or not fully ...

## F2 / f2_tfidf_informed_language_035

- 协议：`incremental_curriculum`
- 词数保留率：`0.811`
- 活跃 TF-IDF 特征保留率：`0.509`
- bot 概率下降：`0.311`
- 实际匹配删除词项：`campaign material; gaza health; ukraine war; feels like; addiction; plainly; instead; numbers; global; family; sober; turns; just; grim`

原文节选：

> Maybe I’m missing something, but how is this Russia-Ukraine war news? Feels like the tag’s just wrong. That’s my gripe as well. The 3 weeks part is bad enough on its own, but if they’re comparing Gaza health ministry numbers to yearly global verified counts elsewhere, they need to just say that plainly in the article. Otherwise the whole thread turns into people arguing over the wording instead of the dead kids, which is grim. Using someone’s addiction as campaign material is grim, honestly. You can dislike the family and ...

删除后节选：

> Maybe I’m missing something, but how is this Russia- news? the tag’s wrong. That’s my gripe as well. The 3 weeks part is bad enough on its own, but if they’re comparing ministry to yearly verified counts elsewhere, they need to say that in the article. Otherwise the whole thread into people arguing over the wording of the dead kids, which is . Using someone’s as is , honestly. You can dislike the and still admit dragging a person for getting is low.

## F2 / f2_tfidf_informed_language_002

- 协议：`self_family`
- 词数保留率：`0.718`
- 活跃 TF-IDF 特征保留率：`0.371`
- bot 概率下降：`0.302`
- 实际匹配删除词项：`somebody; honestly; exactly; mystery; ordered; urgent; public; phrase; cities; expect; starts; bleak; apart; like; grim; yeah; wall; look; bit; low`

原文节选：

> Yeah, "less urgent" sounds like a desk phrase, sorry. If cities are still getting hit, what exactly got less urgent here apart from US stock running low? Six years and still no mastermind named? That's grim, honestly. For a case this public you'd expect more than that by now. Yeah, that’s the bit. If nobody can name who ordered it after six years, it starts to look less like a mystery and more like somebody hit a wall on purpose. Pretty bleak either way.

删除后节选：

> , "less " sounds a desk , sorry. If are still getting hit, what got less here from US stock running ? Six years and still no mastermind named? That's , . For a case this you'd more than that by now. , that’s the . If nobody can name who it after six years, it to less a and more hit a on purpose. Pretty either way.

## F3 / f3_human_anchor_local_tfidf_006

- 协议：`leave_one_family_out`
- 词数保留率：`0.871`
- 活跃 TF-IDF 特征保留率：`0.611`
- bot 概率下降：`0.247`
- 实际匹配删除词项：`substance abuse; state starts; battlefield; suddenly; somebody; dragging; matters; updates; problem; issues; shift; stuff; makes; point; line; ugly; like`

原文节选：

> This part should scare people more than the battlefield updates. When the state starts dragging in people at home over stuff they said, that's a rotten line to cross. I get why people are furious right now, few paragraphs later and all anyone hears is revenge. But a country does not stay normal by making people lower their voice and watch every word. Some people absolutely are doing that. Not everyone, but you can see the shift now: substance abuse quits being treated like an illness and suddenly it becomes, ...

删除后节选：

> This part should scare people more than the . When the in people at home over they said, that's a rotten to cross. I get why people are furious right now, few paragraphs later and all anyone hears is revenge. But a country does not stay normal by making people lower their voice and watch every word. Some people absolutely are doing that. Not everyone, but you can see the now: quits being treated an illness and it becomes, "there you go, he's trash." I get your main though. If ...

## F3 / f3_human_anchor_local_tfidf_018

- 协议：`incremental_curriculum`
- 词数保留率：`0.884`
- 活跃 TF-IDF 特征保留率：`0.759`
- bot 概率下降：`0.251`
- 实际匹配删除词项：`dead kids; pressure; strategy; families; exactly; matters; plainly; figures; numbers; actual; urgent; tragic; argue; cycle; exact; kids; yeah; just; does`

原文节选：

> If that figure is even half right, it's horrific. Argue over the tally all day if you want. A heap of dead kids is not a paperwork dispute. And yeah, spare me the automatic Hamas reply. Everyone knows Hamas exists. It still doesn't make this okay. So what is this meant to be now exactly? From here it's whole families getting erased week after week. Calling that a strategy is obscene. That's the bit where they lose me. "Urgent" sounds great in a statement, but what does it mean by ...

删除后节选：

> If that figure is even half right, it's horrific. over the tally all day if you want. A heap of is not a paperwork dispute. And , spare me the automatic Hamas reply. Everyone knows Hamas exists. It still doesn't make this okay. So what is this meant to be now ? From here it's whole getting erased week after week. Calling that a is obscene. That's the bit where they lose me. " " sounds great in a statement, but what it mean by tomorrow morning? More aid trucks ...

## F4 / f4_peba_trait_prompt_021

- 协议：`incremental_curriculum`
- 词数保留率：`0.814`
- 活跃 TF-IDF 特征保留率：`0.603`
- bot 概率下降：`0.385`
- 实际匹配删除词项：`aid access; addiction; families; partisan; recovery; ukraine; relapse; turning; actual; urgent; action; issue; sober; turns; moral; yeah; ugly; like; real`

原文节选：

> Yeah, “urgent call for action” needs to cash out into actual words. Ceasefire, aid access, fuel, medical corridors, hostage release. Otherwise it turns into the usual moral throat-clearing while children keep getting pulled out of rubble and hospitals keep collapsing. Bleak, but that is the point. Addiction is a medical issue, not a campaign prop. You do not have to like Hunter Biden to grasp that turning relapse and recovery into partisan entertainment poisons how regular people get treated too. And yes, that spills over into families trying to get ...

删除后节选：

> , “ call for ” needs to cash out into words. Ceasefire, , fuel, medical corridors, hostage release. Otherwise it into the usual throat-clearing while children keep getting pulled out of rubble and hospitals keep collapsing. Bleak, but that is the point. is a medical , not a campaign prop. You do not have to Hunter Biden to grasp that and into entertainment poisons how regular people get treated too. And yes, that spills over into trying to get someone . stuff. This looks the wrong tab won. child displacement ...

## F4 / f4_peba_trait_prompt_048

- 协议：`incremental_curriculum`
- 词数保留率：`0.795`
- 活跃 TF-IDF 特征保留率：`0.523`
- bot 概率下降：`0.386`
- 实际匹配删除词项：`global conflict; comparison; correction; headline; horrific; exactly; wording; matters; numbers; arguing; deaths; change; claim; 2019; kids; 300; isn`

原文节选：

> Can we keep the word preliminary doing some work here? 100 to 300 killed is still horrific, but it is not the same as the first 500 claim people repeated everywhere. Isn’t that exactly it? Small correction: knocking 500 down to 100 to 300 is not nitpicking, it is a major factual change. Still horrific, obviously, but early bad numbers always seem to outrun the correction. What exactly is the comparison here? Small correction: when a headline says more than all global conflict annually since 2019, the method matters a ...

删除后节选：

> Can we keep the word preliminary doing some work here? 100 to killed is still , but it is not the same as the first 500 people repeated everywhere. ’t that it? Small : knocking 500 down to 100 to is not nitpicking, it is a major factual . Still , obviously, but early bad always seem to outrun the . What is the here? Small : when a says more than all annually since , the method a lot. The are either way, but if the baseline is fuzzy ...

## F5 / f5_gemini_trait_prompt_021

- 协议：`incremental_curriculum`
- 词数保留率：`0.867`
- 活跃 TF-IDF 特征保留率：`0.659`
- bot 概率下降：`0.308`
- 实际匹配删除词项：`interceptors; getting hit; diplomacy; timeline; dragging; treating; argument; matters; reality; urgent; global; people; shifts; shift; chess; just; like; high; isn`

原文节选：

> Most people aren't tuning into this debate to worry about global diplomacy; they want a concrete timeline on when daily costs will actually stabilize. I get that foreign policy matters in the grand scheme of things. Just to be accurate, though, treating the whole thing like a high-level chess match completely ignores that the average person is just trying to make it through their next week of shifts. Telling people the need isn't urgent completely ignores the day-to-day reality of their power grid getting hit. I know the brass has ...

删除后节选：

> Most aren't tuning into this debate to worry about ; they want a concrete on when daily costs will actually stabilize. I get that foreign policy in the grand scheme of things. to be accurate, though, the whole thing a -level match completely ignores that the average person is trying to make it through their next week of . Telling the need 't completely ignores the day-to-day of their power grid . I know the brass has to audit our own stockpiles before signing off on more transfers; to be ...

## F5 / f5_gemini_trait_prompt_031

- 协议：`incremental_curriculum`
- 词数保留率：`0.872`
- 活跃 TF-IDF 特征保留率：`0.687`
- bot 概率下降：`0.309`
- 实际匹配删除词项：`addiction; civilians; situation; different; recovery; reality; numbers; tragedy; totally; family; feels; exact; awful; just; like; kind; rush; yeah; isn`

原文节选：

> It's definitely an awful situation for the civilians. But are we totally sure about these exact numbers yet? Just going off what my family has seen with deployments, early casualty reports from a combat zone are usually kind of all over the place. Not saying it isn't a tragedy, just maybe we should wait for some independent verification before jumping to conclusions? I sympathize with anyone battling addiction, it's a rough road. But I'm not sure people are actually weaponizing his recovery? Most of the pushback seems to be about ...

删除后节选：

> It's definitely an for the . But are we sure about these yet? going off what my has seen with deployments, early casualty reports from a combat zone are usually of all over the place. Not saying it 't a , maybe we should wait for some independent verification before jumping to conclusions? I sympathize with anyone battling , it's a rough road. But I'm not sure people are actually weaponizing his ? Most of the pushback seems to be about the overseas business deals and the gun form. Seems ...
