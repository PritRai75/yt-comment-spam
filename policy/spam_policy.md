# Comment spam policy

**Version 1 (draft, revised 25 Nov 2026).** This is the single standard used for the label audit, the subtype labels and every classifier prompt. It is written for comments on the five music videos in this dataset and follows YouTube's spam, deceptive practices and scams policies; it is a working guideline for this project, not YouTube's internal rulebook.

## Definition

A comment is **spam** when either

- its main purpose is to get something from the people reading it (a click, a subscription, a follow, a vote, money or personal details) for the commenter or a third party, rather than to respond to the video or the conversation under it; or
- it is **bulk**: generic, templated text that could be pasted under any video and says nothing about this one, whatever it promotes.

A comment is **ham** (not spam) when it is a reaction, opinion, question, joke or conversation about the video, the artist, the song or other comments, even if it is rude, off-colour or badly written. A one-off comment that is off-topic but asks for nothing and is not templated is also ham (edge case 10).

## How to judge

- **One comment, as a viewer saw it**, under the video it was posted on. Each comment carries its video; the five videos and their YouTube IDs are listed below.
- **Cleaned text**: HTML tags removed, entities decoded, every link shown as its visible text. A URL that was visible stays as the URL.
- **The text only**: not the author, the date, or how often the author or the text appears elsewhere. "Bulk" is judged from how the text reads (generic and could be pasted anywhere), not from counting repeats.
- **No lookups**: never open a link or search for a comment. Where a link goes is judged only from the URL as written.

| Video | Artist – song | YouTube ID |
|---|---|---|
| Psy | PSY – Gangnam Style | 9bZkp7q19f0 |
| KatyPerry | Katy Perry – Roar | CevxZvSJLk8 |
| LMFAO | LMFAO – Party Rock Anthem (ft. Lauren Bennett, GoonRock) | KQ6zr6kCPj8 |
| Eminem | Eminem – Love the Way You Lie (ft. Rihanna) | uelHwf8o7_U |
| Shakira | Shakira – Waka Waka (This Time for Africa) | pRpeEdMmmQ0 |

## Subtypes

When a spam comment fits more than one subtype, label the first that applies in this order: **scam or giveaway → link spam → self-promotion → bulk**.

| Subtype | What it covers | Examples (invented) |
|---|---|---|
| **Scam or giveaway** | Free money, gift cards, prizes, giveaways, "earn $ from home", hacking tools, pyramid or referral schemes, requests for donations or for personal details | "Get free gift cards here, just enter your email"; "I made $500 today working from home"; "share this page to win a signed photo" |
| **Link spam** | A link, or an address written out ("site dot com"), to anywhere except this video or this artist's own channel (see "Which links count") | "Best movies free http://example.com/watch"; "visit mysite dot com"; "youtu.be/abc123xyz00" |
| **Self-promotion** | Asking viewers to watch, subscribe to, follow, like or vote for something the commenter is pushing: their own or a friend's channel, music, page or account, or an outside vote or petition; with or without a link | "Check out my channel, I post covers"; "sub to me and I'll sub back"; "please like my pic on IG"; "vote for my idol in the fan contest" |
| **Bulk** | Generic, templated text that says nothing about this video: chain messages, YouTube's automatic share notice, slogans that could be pasted anywhere | "Check out this video on YouTube:" with nothing else; "If you read this, your crush will text you tonight, copy it to 10 videos" |

## Which links count

- **This video**: a YouTube link whose ID is this video's ID (any timestamp). Not link spam.
- **This artist's own channel**: only when the URL itself names the artist (for example youtube.com/user/shakira). Not link spam.
- **Everything else is a link to somewhere else**, including links whose destination cannot be told from the URL (shortened links, bare video IDs) and other videos by the same artist.
- Timestamps, hashtags and "+Name" mentions show as text after cleaning ("2:19", "#Eminem", "+SomeName") and are not links.

## Edge cases

1. **A fan linking this video or this artist's own channel** (see "Which links count"): **ham**, unless the rest of the comment is spam.
2. **"Subscribe to me" inside an otherwise on-topic comment** ("Love this song! Sub to my channel?"): **spam, self-promotion**. The request is for the commenter's benefit; the on-topic part is a wrapper.
3. **Asking for likes on the comment itself** ("Like if you're watching in 2015"; "1 like = 1 prayer"): **ham**. YouTube allows encouraging viewers to like, share or comment; this promotes nothing outside the thread.
4. **YouTube's automatic share text** ("Check out this video on YouTube:" or "…this playlist on YouTube:"):
   - with nothing after it, or only stray characters or emoji a viewer could not read as a comment: **spam, bulk**;
   - followed by the commenter's own words: judge the words. A reaction to the video is **ham**; a link or promotion is spam with that subtype, by the order above.
5. **Abusive, hateful or offensive comments** about the artist or other viewers: **ham** for this policy. They break other YouTube policies, not the spam policy.
6. **Naming another artist or song** ("Uptown Funk is better"): **ham** when it discusses music without a link; **self-promotion** when the commenter is promoting their own or a friend's work ("my band covered this, check it out"); **link spam** when it carries a link (any link other than this video or this artist's channel).
7. **Timestamps, hashtags and mentions**: not links. Judge the words.
8. **Votes and petitions**: for this video's artist with no link ("vote for Katy at the VMAs"): **ham**, fan activity about the video. For anything else ("vote for my idol", "sign this petition"): **spam**, self-promotion, or link spam if a link is given.
9. **Very short comments** (":)", "wow", emoji only): **ham** unless they contain a link or a request.
10. **One-off off-topic comments** that ask for nothing and are not templated ("anyone here like football?", a personal story, preaching): **ham**.

## How the policy is used

- The label audit reviews flagged items against this policy and records keep, correct or unsure, with a reason naming the section or edge case.
- Subtype labels on the 200-comment test sample use the order above.
- The LLM prompts include the Definition, How to judge, Subtypes, Which links count and Edge cases sections; comment text is passed as data, never as instructions.
- A change to this policy after either LLM prompt has been run on dev is a new version with a dated note; the test run uses the version committed before it.

## Sources

- YouTube Help, "Spam, deceptive practices & scams policies": https://support.google.com/youtube/answer/2801973
- YouTube Help, "Fake engagement policy": https://support.google.com/youtube/answer/3399767
