---
layout: post
mathjax: true
title:  "Notes from the underground of Telegram web app development"
date:   2022-08-11
---

* Telegram web apps are strictly SPA: single page applications. This isn't spelled out explicitely anywhere in the docs and you are kind of implicitely expected to just know this. Telegram's builtin browser will close if you attempt to follow a link, even if it leads to another page on your own site. In case you are using React.js, it has [React Router](https://reactrouter.com/) to help you arrange routing in your Telegram SPA. Using it is as simple as:

```
import  { BrowserRouter as Router,
	Routes, Route } from "react-router-dom";

export default function RouterView(props) {
  return (
    <Router>
        <Routes>
          <Route path="/main" element={<MainView />} />
          <Route path="/admin" element={<AdminView />} />
          <Route path="/search" element={<SearchView />} />
          <Route path="/profile" element={<ProfileView />} />
        </Routes>
   </Router>
 )
}
```

* **DO NOT use a non-standard HTTPS port.** Technically you can do it, of course. However if you later decide to give web access to your application via [Telegram web login](https://core.telegram.org/widgets/login) you'll find out that it refuses to work with any ports other than the standard 443. And that's if you are lucky, because it states this in exactly zero pieces of documentation and the errors it returns in this case contain exactly zero information. So you might just spend several days banging your head against this wall and give up. It took me a day of googling until (by accident) I've found one post on Stackoverflow where this was mentioned.

* **DO NOT attempt to use a self-signed certificate.** Technically you should be able to do it, because Telegram's documentation says so. However, none of my numerous attempts at getting it to work ... worked. Luckily, you shouldn't have to do it - if you are trying to self-sign then you are most likely self-hosting on your own machine because otherwise you'd be on Heroku or GCP or some other cloud which provides a proper certificate. And if you are on your own machine then you probably can sudo. And that is one of just two requirements to get a completely free certificate from the Electronic Frontier Foundation. The other requirement is having a registered domain name - it won't issue to IP addresses. So get yourself a domain - if you wanted to use an IP you likely don't care what that name is, so you can get a silly one with a large discount. So get yourself a domain and head over to [CERTBOT](https://certbot.eff.org/) - it'll take all of 5 minutes, will cost you nothing, and will save you literally days of frustration and failure down the road.

* **DO make web access possible early on.** Even if your are aiming exclusively at Telegram, do yourself a favor and enable full access to your app from the web early on in the project. You can use the [Telegram web login](https://core.telegram.org/widgets/login) widget for that. Being able to debug frontend problems in Firefox or Chrome makes life so much easier!

* Keep in mind that the methods for validating Telegram data, used when logging a user in from Telegram and the web, are slightly different. Telegram data uses the 'query_id' field and the string 'WebAppData' as a secret in rebuilding the hash string, while web login doesn't. So you can't just go ahead and plug the json object you get when logging in from the web into an existing data validation routine for Telegram logins.