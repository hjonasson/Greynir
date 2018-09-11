
/*

   Reynir: Natural language processing for Icelandic

   Page.js

   Scripts for displaying tokenized and parsed text,
   with pop-up tags on hover, name registry, statistics, etc.

   Copyright (C) 2018 Miðeind ehf.
   Author: Vilhjálmur Þorsteinsson
   All rights reserved

      This program is free software: you can redistribute it and/or modify
      it under the terms of the GNU General Public License as published by
      the Free Software Foundation, either version 3 of the License, or
      (at your option) any later version.
      This program is distributed in the hope that it will be useful,
      but WITHOUT ANY WARRANTY; without even the implied warranty of
      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
      GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see http://www.gnu.org/licenses/.


   For details about the token JSON format, see TreeUtility.dump_tokens() in treeutil.py.
   t.x is original token text.
   t.k is the token kind (TOK_x). If omitted, this is TOK_WORD.
   t.t is the name of the matching terminal, if any.
   t.m is the BÍN meaning of the token, if any, as a tuple as follows:
      t.m[0] is the lemma (stofn)
      t.m[1] is the word category (ordfl)
      t.m[2] is the word subcategory (fl)
      t.m[3] is the word meaning/declination (beyging)
   t.v contains auxiliary information, depending on the token kind
   t.err is 1 if the token is an error token

*/

// Punctuation types

var TP_LEFT = 1;
var TP_CENTER = 2;
var TP_RIGHT = 3;
var TP_NONE = 4; // Tight - no whitespace around
var TP_WORD = 5;

// Token spacing

var TP_SPACE = [
    // Next token is:
    // LEFT    CENTER  RIGHT   NONE    WORD
    // Last token was TP_LEFT:
    [ false,  true,   false,  false,  false],
    // Last token was TP_CENTER:
    [ true,   true,   true,   true,   true],
    // Last token was TP_RIGHT:
    [ true,   true,   false,  false,  true],
    // Last token was TP_NONE:
    [ false,  true,   false,  false,  false],
    // Last token was TP_WORD:
    [ true,   true,   false,  false,  true]
];

var LEFT_PUNCTUATION = "([„‚«#$€<°";
var RIGHT_PUNCTUATION = ".,:;)]!%?“»”’‛‘…>–";
var NONE_PUNCTUATION = "—–-/'~\\";
// CENTER_PUNCTUATION = '"*&+=@©|'

// Words array
var w = [];

// Name dictionary
var nameDict = { };

function debugMode() {
   return false;
}

function spacing(t) {
   // Determine the spacing requirements of a token
   if (t.k != TOK_PUNCTUATION)
      return TP_WORD;
   if (LEFT_PUNCTUATION.indexOf(t.x) > -1)
      return TP_LEFT;
   if (RIGHT_PUNCTUATION.indexOf(t.x) > -1)
      return TP_RIGHT;
   if (NONE_PUNCTUATION.indexOf(t.x) > -1)
      return TP_NONE;
   return TP_CENTER;
}

function queryPerson(name) {
   // Navigate to the main page with a person query
   window.location.href = "/?f=q&q=" + encodeURIComponent("Hver er " + name + "?");
}

function queryEntity(name) {
   // Navigate to the main page with an entity query
   window.location.href = "/?f=q&q=" + encodeURIComponent("Hvað er " + name + "?");
}

function showParse(ev) {
   // A sentence has been clicked: show its parse grid
   var sentText = $(ev.delegateTarget).text();
   // Do an HTML POST to the parsegrid URL, passing
   // the sentence text within a synthetic form
   // serverPost("/parsegrid", { txt: sentText, debug: debugMode() }, false)
   window.location.href = "/treegrid?txt=" + encodeURIComponent(sentText);
}

function showPerson(ev) {
   // Send a query to the server
   var name = undefined;
   var wId = $(this).attr("id"); // Check for token id
   if (wId !== undefined) {
      // Obtain the name in nominative case from the token
      var ix = parseInt(wId.slice(1));
      if (w[ix] !== undefined)
         name = w[ix].v;
   }
   if (name === undefined)
      name = $(this).text(); // No associated token: use the contained text
   queryPerson(name);
   ev.stopPropagation();
}

function showEntity(ev) {
   // Send a query to the server
   var ename = $(this).text();
   var nd = nameDict[ename];
   if (nd && nd.kind == "ref")
      // Last name reference to a full name entity
      // ('Clinton' -> 'Hillary Rodham Clinton')
      // In this case, we assume that we're asking about a person
      queryPerson(nd.fullname);
   else
      queryEntity(ename);
   ev.stopPropagation();
}

function hoverIn() {
   // Hovering over a token
   var wId = $(this).attr("id");
   if (wId === null || wId === undefined)
      // No id: nothing to do
      return;
   var ix = parseInt(wId.slice(1));
   var t = w[ix];
   if (!t)
      // No token: nothing to do
      return;

   // Save our position
   var offset = $(this).position();
   // Highlight the token
   $(this).addClass("highlight");

   var r = tokenInfo(t, nameDict);

   if (!r.grammar && !r.lemma && !r.details)
      // Nothing interesting to show (probably the sentence didn't parse)
      return;

   $("#grammar").html(r.grammar || "");
   $("#lemma").text(r.lemma || "");
   $("#details").text(r.details || "");

   // Display the percentage bar if we have percentage info
   if (r.percent !== null)
      makePercentGraph(r.percent);
   else
      $("#percent").css("display", "none");

   $("#info").removeClass();
   if (r.class !== null)
      $("#info").addClass(r.class);

   $("#info span#tag")
      .removeClass()
      .addClass("glyphicon")
      .addClass(r.tagClass ? r.tagClass : "glyphicon-tag");

   // Position the info popup
   $("#info")
      .css("top", offset.top.toString() + "px")
      .css("left", offset.left.toString() + "px")
      .css("visibility", "visible");
}

function hoverOut() {
   // Stop hovering over a word
   $("#info").css("visibility", "hidden");
   $(this).removeClass("highlight");
   var wId = $(this).attr("id");
   if (wId === null || wId === undefined)
      // No id: nothing more to do
      return;
   var ix = parseInt(wId.slice(1));
   var t = w[ix];
   if (!t)
      return;
}

function displayTokens(j) {
   var x = ""; // Result text
   var lastSp;
   w = [];
   if (j !== null)
      $.each(j, function(pix, p) {
         // Paragraph p
         x += "<p>\n";
         $.each(p, function(six, s) {
            // Sentence s
            var err = false;
            lastSp = TP_NONE;
            // Check whether the sentence has an error or was fully parsed
            $.each(s, function(tix, t) {
               if (t.err == 1) {
                  err = true;
                  return false; // Break the iteration
               }
            });
            if (err)
               x += "<span class='sent err'>";
            else
               x += "<span class='sent parsed'>";
            $.each(s, function(tix, t) {
               // Token t
               var thisSp = spacing(t);
               // Insert a space in front of this word if required
               // (but never at the start of a sentence)
               if (TP_SPACE[lastSp - 1][thisSp - 1] && tix)
                  x += " ";
               lastSp = thisSp;
               if (t.err)
                  // Mark an error token
                  x += "<span class='errtok'>";
               if (t.k == TOK_PUNCTUATION)
                  x += "<i class='p'>" + ((t.x == "—") ? " — " : t.x) + "</i>"; // Space around em-dash
               else {
                  var cls;
                  var tx = t.x;
                  if (!t.k) {
                     // TOK_WORD
                     if (err)
                        cls = "";
                     else
                     if (t.m)
                        // Word class (noun, verb, adjective...)
                        cls = " class='" + t.m[1] + "'";
                     else
                     if (t.t && t.t.split("_")[0] == "sérnafn") {
                        // Special case to display 'sérnafn' as 'entity'
                        cls = " class='entity'";
                        tx = tx.replace(" - ", "-"); // Tight hyphen, no whitespace
                     }
                     else
                        // Not found
                        cls = " class='nf'";
                  }
                  else {
                     cls = " class='" + tokClass[t.k] + "'";
                     if (t.k == TOK_ENTITY)
                        tx = tx.replace(" - ", "-"); // Tight hyphen, no whitespace
                  }
                  x += "<i id='w" + w.length + "'" + cls + ">" + tx + "</i>";
                  // Append to word/token list
                  w.push(t);
               }
               if (t.err)
                  x += "</span>";
            });
            // Finish sentence
            x += "</span>\n";
         });
         // Finish paragraph
         x += "</p>\n";
      });
   // Show the page text
   $("div#result").html(x);
   // Put a hover handler on each word
   $("div#result i").hover(hoverIn, hoverOut);
   // Put a click handler on each sentence
   $("span.sent").click(showParse);
   // Separate click handler on entity names
   $("i.entity").click(showEntity);
   // Separate click handler on person names
   $("i.person").click(showPerson);
}

function populateStats(stats) {
   $("#tok-num").text(format_is(stats.num_tokens));
   $("#num-sent").text(format_is(stats.num_sentences));
   $("#num-parsed-sent").text(format_is(stats.num_parsed));
   if (stats.num_sentences > 0)
      $("#num-parsed-ratio").text(format_is(100.0 * stats.num_parsed / stats.num_sentences, 1));
   else
      $("#num-parsed-ratio").text("0.0");
   $("#avg-ambig-factor").text(format_is(stats.ambiguity, 2));
   $("div#statistics").css("display", "block");
}

function populateRegister() {
   // Populate the name register display
   var i, item, name, title;
   var register = [];
   $("#namelist").html("");
   $.each(nameDict, function(name, desc) {
      // kind is 'ref', 'name' or 'entity'
      if (desc.kind != "ref")
         // We don't display references to full names
         // Whitespace around hyphens is eliminated for display
         register.push({ name: name.replace(" - ", "-"), title: desc.title, kind: desc.kind });
   });
   register.sort(function(a, b) {
      return a.name.localeCompare(b.name);
   });
   for (i = 0; i < register.length; i++) {
      var ri = register[i];
      item = $("<li></li>");
      name = $("<span></span>").addClass(ri.kind).text(ri.name);
      title = $("<span></span>").addClass("title").text(ri.title);
      item.append(name);
      item.append(title);
      $("#namelist").append(item);
   }
   // Display the register
   if (register.length) {
      $("#register").css("display", "block");
      $("#namelist span.name").click(function(ev) {
         // Send a person query to the server
         queryPerson($(this).text());
      });
      $("#namelist span.entity").click(function(ev) {
         // Send an entity query to the server
         queryEntity($(this).text());
      });
   }
}

