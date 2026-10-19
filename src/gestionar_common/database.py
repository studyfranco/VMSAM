"""Database helpers shared by the show and movie packages and by main_gestionar.

Nothing here imports a gestionar package, so the movie public API can use it
without pulling the show's package.
"""

from sqlalchemy import create_engine

def get_connect_args(database_url):
    connect_args = {}
    if database_url.startswith("postgresql"):
        # PostgreSQL-specific argument (psycopg2)
        connect_args['connect_timeout'] = 60
    elif database_url.startswith("sqlite"):
        # SQLite-specific argument
        connect_args['timeout'] = 61
    return connect_args

def is_database_unavailable(error):
    """True when the error means the database could not be reached, not that a query was wrong.

    OperationalError covers a refused or dropped connection and a server in
    recovery mode; a DBAPIError flagged connection_invalidated is a connection
    that died under a running statement.
    """
    from sqlalchemy.exc import OperationalError, InterfaceError, DBAPIError
    if isinstance(error, (OperationalError, InterfaceError)):
        return True
    return isinstance(error, DBAPIError) and error.connection_invalidated

def wait_for_database(database_url, first_delay=5, max_delay=60):
    """Block until the database answers `SELECT 1`, then return.

    A network cut or a PostgreSQL restart (recovery mode, "not yet accepting
    connections") refuses every connection immediately, so without this wait a
    loop burns through its work and logs one error per item. The delay doubles
    from first_delay up to max_delay; one line per failed attempt goes to
    stderr, and one when the database is back. Returns the number of failed
    attempts (0 when the database was reachable at once). Only an
    unavailability is waited for: any other error is raised.
    """
    from sqlalchemy import text
    from time import sleep
    from sys import stderr
    delay = first_delay
    failed_attempts = 0
    while True:
        engine = create_engine(database_url, echo=False, connect_args=get_connect_args(database_url))
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            if failed_attempts:
                stderr.write(f"Database reachable again after {failed_attempts} failed attempt(s)\n")
            return failed_attempts
        except Exception as e:
            if not is_database_unavailable(e):
                raise
            failed_attempts += 1
            reason = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
            stderr.write(f"Database unreachable (attempt {failed_attempts}), next try in {delay}s: {reason}\n")
            sleep(delay)
            delay = min(delay*2, max_delay)
        finally:
            engine.dispose()
