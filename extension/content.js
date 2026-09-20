/**
 * Content script — minimal event relay.
 *
 * In Phase 0, the content script does NOT manipulate the DOM (that's Playwright's
 * job via CDP). It only provides a communication bridge and basic page info.
 */

// Relay page info to the background script on load
chrome.runtime.sendMessage({
  type: "page_info",
  url: window.location.href,
  title: document.title,
});

// Listen for any future requests from the background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "get_page_info") {
    sendResponse({
      url: window.location.href,
      title: document.title,
    });
  }
  return true;
});
