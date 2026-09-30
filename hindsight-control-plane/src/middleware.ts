import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import createIntlMiddleware from "next-intl/middleware";

import { ACCESS_KEY_COOKIE, verifySessionToken } from "@/lib/auth/session";
import { stripBasePath, withBasePath } from "@/lib/base-path";
import { routing } from "@/i18n/routing";
import { locales } from "@/i18n/config";

// Routes that don't require authentication
const PUBLIC_PATTERNS = [
  "/login",
  "/api/auth/",
  "/api/health",
  "/api/version",
  "/logo.png",
  "/favicon",
  "/_next",
  "/fonts",
  "/static",
];

const intlMiddleware = createIntlMiddleware(routing);

const LOCALE_PREFIX = new RegExp(`^/(?:${locales.join("|")})(?=/|$)`);

/**
 * Strip a leading locale segment from an app pathname.
 *
 * `routing.localePrefix` is "never", so the locale never appears in the URL —
 * but next-intl still rewrites `/login` to `/en/login` internally, and the
 * middleware runs again on that rewritten path. `stripBasePath` only knows
 * about `basePath`, so without this the public-route check sees `/en/login`,
 * fails `startsWith("/login")`, and bounces the browser back to
 * `/login?returnTo=/en/login` forever.
 */
function stripLocalePrefix(pathname: string): string {
  const stripped = pathname.replace(LOCALE_PREFIX, "");
  return stripped || "/";
}

export async function middleware(request: NextRequest) {
  const accessKey = process.env.HINDSIGHT_CP_ACCESS_KEY;
  const { pathname } = request.nextUrl;
  const appPathname = stripLocalePrefix(stripBasePath(pathname));
  const onRewrittenPass = LOCALE_PREFIX.test(pathname);

  // API routes are not locale-prefixed — handle auth directly without i18n routing.
  if (appPathname.startsWith("/api/")) {
    if (!accessKey) {
      return NextResponse.next();
    }

    const isPublic = PUBLIC_PATTERNS.some((pattern) => appPathname.startsWith(pattern));
    if (isPublic) {
      return NextResponse.next();
    }

    const sessionCookie = request.cookies.get(ACCESS_KEY_COOKIE)?.value;
    const isAuthenticated = await verifySessionToken(sessionCookie, accessKey);

    if (!isAuthenticated) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "Unauthorized",
          errorKey: "api.errors.auth.unauthorized",
        }),
        { status: 401 }
      );
    }

    return NextResponse.next();
  }

  // Page routes: enforce auth first, then delegate to the i18n middleware for
  // locale negotiation and rewriting. With localePrefix "never" the locale is
  // never in the path, so appPathname is already the canonical route.
  if (accessKey) {
    const isPublic = PUBLIC_PATTERNS.some((pattern) => appPathname.startsWith(pattern));

    if (!isPublic) {
      const sessionCookie = request.cookies.get(ACCESS_KEY_COOKIE)?.value;
      const isAuthenticated = await verifySessionToken(sessionCookie, accessKey);

      if (!isAuthenticated) {
        // Next.js middleware redirects do not automatically inherit next.config basePath.
        // Prefix the target explicitly, but keep returnTo as the app-relative path so
        // client-side router.push() does not double-prefix after login.
        const loginUrl = new URL(withBasePath("/login"), request.url);
        loginUrl.searchParams.set("returnTo", appPathname);
        return NextResponse.redirect(loginUrl);
      }
    }
  }

  // `localePrefix: "never"` still rewrites `/login` to `/en/login`, and the
  // middleware runs a second time on that rewritten path. Calling the intl
  // middleware again there is wrong: next-intl sees a locale-prefixed path it
  // would have to strip, answers 307 back to `/login`, and the browser bounces
  // between the two forever (ERR_TOO_MANY_REDIRECTS). The rewrite already
  // happened on the first pass, so the second one just lets the request
  // through. The auth check above still runs, with the locale stripped.
  return onRewrittenPass ? NextResponse.next() : intlMiddleware(request);
}

export const config = {
  // Match all paths except Next.js internals and static assets.
  // - Use an explicit file extension allowlist instead of .*\..* so that
  //   dynamic segments containing dots (e.g. bank IDs like
  //   "SX.Products.GovComply.Build") still get the i18n locale rewrite.
  matcher:
    "/((?!_next|_vercel|.*\\.(?:png|jpe?g|gif|svg|webp|ico|css|js|map|woff2?|ttf|eot|txt|xml|json)$).*)",
};
