def _get_or_create_athlete_id(strava_athlete_id: int, firstname: str | None, lastname: str | None) -> int:
    """
    Upsert по strava_athlete_id и връща вътрешния athletes.id
    """
    with SessionLocal() as s:
        # upsert по уникалния strava_athlete_id
        a_stmt = pg_insert(Athlete.__table__).values({
            "strava_athlete_id": strava_athlete_id,
            "firstname": firstname,
            "lastname": lastname,
        }).on_conflict_do_update(
            index_elements=[Athlete.__table__.c.strava_athlete_id],
            set_={
                "firstname": firstname,
                "lastname": lastname,
            },
        )
        s.execute(a_stmt)
        # вземи вътрешния id
        res = s.execute(
            Athlete.__table__.select().with_only_columns(Athlete.__table__.c.id).where(
                Athlete.__table__.c.strava_athlete_id == strava_athlete_id
            )
        )
        ath_id = res.scalar_one()
        s.commit()
        return int(ath_id)


def upsert_athlete_and_tokens(
    strava_athlete_id: int,
    firstname: str | None,
    lastname: str | None,
    access_token: str,
    refresh_token: str,
    expires_at: int,
    scope: str | None = None,  # игнорираме го, защото колоната липсва
) -> None:
    """
    Upsert на athlete и tokens.
    ПО-ВАЖНО: в strava_tokens.athlete_id записваме ВЪТРЕШНОТО athletes.id (не Strava ID),
    за да спазим външния ключ в базата.
    """
    # 1) осигури athlete ред и вземи вътрешния id
    athlete_internal_id = _get_or_create_athlete_id(strava_athlete_id, firstname, lastname)

    # 2) upsert на токени по athlete_id (= вътрешното athletes.id)
    with SessionLocal() as s:
        t_stmt = pg_insert(StravaToken.__table__).values({
            "athlete_id": athlete_internal_id,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_at": expires_at,
        }).on_conflict_do_update(
            index_elements=[StravaToken.__table__.c.athlete_id],
            set_={
                "access_token": access_token,
                "refresh_token": refresh_token,
                "expires_at": expires_at,
                "updated_at": text("timezone('utc', now())"),
            },
        )
        s.execute(t_stmt)
        s.commit()
